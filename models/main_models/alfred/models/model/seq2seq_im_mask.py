import os
import json
import torch
import numpy as np
import nn.vnn as vnn
import collections
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from model.seq2seq import Module as Base
from models.utils.metric import compute_f1, compute_exact
from gen.utils.image_util import decompress_mask


class Module(Base):

    def __init__(self, args, vocab):
        '''
        Seq2Seq agent
        '''
        super().__init__(args, vocab)

        # encoder and self-attention
        self.enc = nn.LSTM(args.demb, args.dhid, bidirectional=True, batch_first=True)
        self.enc_att = vnn.SelfAttn(args.dhid*2)

        # subgoal monitoring
        # self.subgoal_monitoring = (self.args.pm_aux_loss_wt > 0 or self.args.subgoal_aux_loss_wt > 0)

        # get action max and mins
        max_min_path = os.path.join(self.args.pp_data, 'max_min.json')
        with open(max_min_path, 'r') as f:
            max_min_data = json.load(f)
        max_vals = max_min_data['max']
        min_vals = max_min_data['min']

        # model to be finetuned
        decoder = vnn.ConvFrameMaskDecoder
        self.dec = decoder(max_vals, min_vals, self.emb_action_low, self.args.bins, args.class_mode, args.demb, args.dframe, 2*args.dhid,
                           pframe=args.pframe,
                           attn_dropout=args.attn_dropout,
                           hstate_dropout=args.hstate_dropout,
                           actor_dropout=args.actor_dropout,
                           input_dropout=args.input_dropout,
                           adapter_dropout=args.adapter_dropout,
                           teacher_forcing=args.dec_teacher_forcing,
                           continuous_action_dim = args.continuous_action_dim)
        self.load_pretrained_model(args.finetune)
        self.freeze()
        
        # dropouts
        self.vis_dropout = nn.Dropout(args.vis_dropout)
        self.lang_dropout = nn.Dropout(args.lang_dropout, inplace=True)
        self.input_dropout = nn.Dropout(args.input_dropout)

        # internal states
        self.state_t = None
        self.e_t = None
        self.test_mode = False

        self.mse_loss = torch.nn.MSELoss(reduction='none')

        # paths
        self.root_path = os.getcwd()
        self.feat_pt = 'feat_conv.pt'

        # params
        # self.max_subgoals = 25

        # reset model
        self.reset()

    def freeze(self):
        """
        The layers to be fine-tuned:
            emb_word.weight
            dec.adapter.weight
            dec.adapter.bias
            dec.actor.weight
            dec.actor.bias
        """

        for name, param in self.named_parameters():
            if 'actor' not in name and 'emb_word' not in name and 'adapter' not in name:
                param.requires_grad = False

    def load_pretrained_model(self, path):

         # You might want to filter out unnecessary keys
        model_dict = self.state_dict()

        # Load the pretrained state dict
        pretrained_dict = torch.load(path)

        # Load pretrained model state dictionary
        pretrained_state_dict = pretrained_dict['model']

        #don't want to use actor and emb_word pretrained params so filtering them out
        filtered_dict = {key: value for key, value in pretrained_state_dict.items()
                     if 'actor' not in key and 'emb_word' not in key}
       
        #loads in the keys shared between the current model and the pretrained model while also removing the keys we want to fine-tune
        self.load_state_dict(filtered_dict, strict=False)

    def featurize(self, batch, load_mask=True, load_frames=True):
        '''
        tensorize and pad batch input
        '''
        device = torch.device('cuda') if self.args.gpu else torch.device('cpu')
        feat = collections.defaultdict(list) # for all trajs in the batch

        for ex in batch:
            ###########
            # auxillary
            ###########

            # if not self.test_mode:
                # subgoal completion supervision
                # if self.args.subgoal_aux_loss_wt > 0:
                #     feat['subgoals_completed'].append(np.array(ex['num']['low_to_high_idx']) / self.max_subgoals)

                # progress monitor supervision
                # if self.args.pm_aux_loss_wt > 0:
                #     num_actions = len([a for sg in ex['num']['action_low'] for a in sg])
                #     subgoal_progress = [(i+1)/float(num_actions) for i in range(num_actions)]
                #     feat['subgoal_progress'].append(subgoal_progress)

            #########
            # inputs
            #########

            # goal and instr language
            lang_goal = ex['num']['lang_goal']

            # zero inputs if specified
            lang_goal = self.zero_input(lang_goal) if self.args.zero_goal else lang_goal

            # append goal + instr
            lang_goal_instr = lang_goal
            feat['lang_goal_instr'].append(lang_goal_instr)

            # load Resnet features from disk
            if load_frames and not self.test_mode:
                root = ex['root']
                root = 'data/feats' + root[19:] #delete later maybe
                im = torch.load(os.path.join(root, 'pp', self.feat_pt))

                
                num_low_actions =  len(ex['num']['action_low']) #already has the stop action so len is already +1
                im = torch.cat((im, im[-1].unsqueeze(0)), dim=0) #add one more frame that's a copy of the last frame so len(frames) matches len(actions) due to a stop action being added
                num_feat_frames = im.shape[0]

                # Modeling Quickstart (without filler frames)
                if num_low_actions == num_feat_frames:
                    feat['frames'].append(im)

                # Full Dataset (contains filler frames)
                #won't run for ours since every frame is accompanied by an action
                else:
                    keep = [None] * num_low_actions
                    for i, d in enumerate(ex['images']):
                        # only add frames linked with low-level actions (i.e. skip filler frames like smooth rotations and dish washing)
                        if keep[d['low_idx']] is None:
                            keep[d['low_idx']] = im[i]
                    keep[-1] = im[-1]  # stop frame
                    feat['frames'].append(torch.stack(keep, dim=0))

            #########
            # outputs
            #########

            if not self.test_mode:
                # low-level action
                feat['action_low'].append(ex['num']['action_low']) #append trajectory's sequence of actions. feat['action_low'] should end up being a list that's batch num long

        # tensorization and padding
        for k, v in feat.items():
            if k in {'lang_goal_instr'}:
                # language embedding and padding
                seqs = [torch.tensor(vv, device=device) for vv in v]
                pad_seq = pad_sequence(seqs, batch_first=True, padding_value=self.pad)
                seq_lengths = np.array(list(map(len, v)))
                embed_seq = self.emb_word(pad_seq)
                packed_input = pack_padded_sequence(embed_seq, seq_lengths, batch_first=True, enforce_sorted=False)
                feat[k] = packed_input
            else:
                # default: tensorize and pad sequence

                seqs = [vv.clone().detach().to(device=device, dtype=torch.float) if 'frames' in k else 
                                [{key: torch.tensor(value, device=device, dtype=torch.int) for key, value in d.items()} for d in vv] 
                                for vv in v]
                if k in {'action_low'}:
                #seqs is list of length batch where each item is a list that's traj length of dictionaries that contain actions where the actions are float tensors

                    # Determine the maximum length of any list in the seqs
                    max_length = max(len(lst) for lst in seqs)

                    template_dict = {
                        'state_body': torch.full((4,), 1),
                        'state_ee': torch.full((6,), 1) 
                    }

                    # Pad each list in seqs to the maximum length
                    pad_seq = [
                        lst + [template_dict.copy() for _ in range(max_length - len(lst))] for lst in seqs
                    ]
                else:
                    pad_seq = pad_sequence(seqs, batch_first=True, padding_value=self.pad)
            
                feat[k] = pad_seq
        return feat


    def forward(self, feat, max_decode=300):
        cont_lang, enc_lang = self.encode_lang(feat)
        state_0 = cont_lang, torch.zeros_like(cont_lang) #self-attention encoding & 0-tensor with same len for every traj in the batch as an init hidden decoding
        frames = self.vis_dropout(feat['frames'])
        res = self.dec(enc_lang, frames, max_decode=max_decode, gold=feat['action_low'], state_0=state_0)
        feat.update(res)
        return feat


    def encode_lang(self, feat):
        '''
        encode goal+instr language
        '''
        emb_lang_goal_instr = feat['lang_goal_instr']
        self.lang_dropout(emb_lang_goal_instr.data)
        enc_lang_goal_instr, _ = self.enc(emb_lang_goal_instr) #LSTM encoding
        enc_lang_goal_instr, _ = pad_packed_sequence(enc_lang_goal_instr, batch_first=True)
        self.lang_dropout(enc_lang_goal_instr)
        cont_lang_goal_instr = self.enc_att(enc_lang_goal_instr) #self-attention encoding

        return cont_lang_goal_instr, enc_lang_goal_instr


    def reset(self):
        '''
        reset internal states (used for real-time execution during eval)
        '''
        self.r_state = {
            'state_t': None,
            'e_t': None,
            'cont_lang': None,
            'enc_lang': None
        }

    def step(self, feat, prev_action=None):
        '''
        forward the model for a single time-step (used for real-time execution during eval)
        '''

        # encode language features
        if self.r_state['cont_lang'] is None and self.r_state['enc_lang'] is None:
            self.r_state['cont_lang'], self.r_state['enc_lang'] = self.encode_lang(feat)

        # initialize embedding and hidden states
        if self.r_state['e_t'] is None and self.r_state['state_t'] is None:
            self.r_state['e_t'] = self.dec.go.repeat(self.r_state['enc_lang'].size(0), 1)
            self.r_state['state_t'] = self.r_state['cont_lang'], torch.zeros_like(self.r_state['cont_lang'])

        # previous action embedding
        e_t = self.embed_action(prev_action) if prev_action is not None else self.r_state['e_t']

        # decode and save embedding and hidden states
        out_action_low, out_action_low_mask, state_t, *_ = self.dec.step(self.r_state['enc_lang'], feat['frames'][:, 0], e_t=e_t, state_tm1=self.r_state['state_t'])

        # save states
        self.r_state['state_t'] = state_t
        self.r_state['e_t'] = self.dec.emb(out_action_low.max(1)[1])

        # output formatting
        feat['out_action_low'] = out_action_low.unsqueeze(0)
        feat['out_action_low_mask'] = out_action_low_mask.unsqueeze(0)
        return feat


    def extract_preds(self, out, batch, feat, clean_special_tokens=True):
        '''
        output processing
        '''
        pred = {}
        for ex, alow, alow_mask in zip(batch, feat['out_action_low'].max(2)[1].tolist(), feat['out_action_low_mask']):
            # remove padding tokens
            if self.pad in alow:
                pad_start_idx = alow.index(self.pad)
                alow = alow[:pad_start_idx]
                alow_mask = alow_mask[:pad_start_idx]

            if clean_special_tokens:
                # remove <<stop>> tokens
                if self.stop_token in alow:
                    stop_start_idx = alow.index(self.stop_token)
                    alow = alow[:stop_start_idx]
                    alow_mask = alow_mask[:stop_start_idx]

            # index to API actions
            words = self.vocab['action_low'].index2word(alow)

            # sigmoid preds to binary mask
            alow_mask = F.sigmoid(alow_mask)
            p_mask = [(alow_mask[t] > 0.5).cpu().numpy() for t in range(alow_mask.shape[0])]

            task_id_ann = self.get_task_and_ann_id(ex)
            pred[task_id_ann] = {
                'action_low': ' '.join(words),
                'action_low_mask': p_mask,
            }

        return pred


    def embed_action(self, action):
        '''
        embed low-level action
        '''
        device = torch.device('cuda') if self.args.gpu else torch.device('cpu')
        action_num = torch.tensor(self.vocab['action_low'].word2index(action), device=device)
        action_emb = self.dec.emb(action_num).unsqueeze(0)
        return action_emb


    def compute_loss(self, out, batch, feat):
        '''
        loss function for Seq2Seq agent
        '''

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        losses = dict()

        # GT and predictions
        if self.args.class_mode:
            p_alow = out['out_action_low'].flatten(0, 1)
        else:
            p_alow = out['out_action_low'].view(-1, self.args.continuous_action_dim)
        
        l_alow = [torch.cat([item['state_body'].to(device), item['state_ee'].to(device)]) for sublist in feat['action_low'] for item in sublist]
        l_alow = torch.stack(l_alow)

        # action loss
        pad_tensor = torch.full_like(l_alow, 1) #1 is the action pad index for class mode
        pad_valid = (l_alow != pad_tensor).all(dim=1) #collapse the bools in the inner tensors to 1 bool
        if self.args.class_mode:
            total_loss = torch.zeros(l_alow.shape[0]).to('cuda')
            for dim in range(p_alow.shape[1]): #loops 10 times, one for each action dim
                loss = nn.CrossEntropyLoss(reduction='none')(p_alow[:, dim, :], l_alow[:, dim])
                total_loss += loss #add all action dims losses together for each trajectory
            alow_loss = total_loss / l_alow.shape[1] #avg loss for all action dims losses for each trajectory
        else:
            alow_loss = F.mse_loss(p_alow, l_alow, reduction='none')
        # Apply the validity mask to the loss tensor
        alow_loss *= pad_valid.float()
        # Calculate the mean loss only over valid elements
        valid_loss_sum = alow_loss.sum()
        valid_count = pad_valid.float().sum()
        alow_loss_mean = valid_loss_sum / valid_count
        losses['action_low'] = alow_loss_mean * self.args.action_loss_wt
        
        return losses

    def flip_tensor(self, tensor, on_zero=1, on_non_zero=0):
        '''
        flip 0 and 1 values in tensor
        '''
        res = tensor.clone()
        res[tensor == 0] = on_zero
        res[tensor != 0] = on_non_zero
        return res


    def compute_metric(self, preds, data):
        '''
        compute f1 and extract match scores for output
        '''
        m = collections.defaultdict(list)
        for task in data:
            ex = self.load_task_json(task)
            i = self.get_task_and_ann_id(ex)
            label = ' '.join([a['discrete_action']['action'] for a in ex['plan']['low_actions']])
            m['action_low_f1'].append(compute_f1(label.lower(), preds[i]['action_low'].lower()))
            m['action_low_em'].append(compute_exact(label.lower(), preds[i]['action_low'].lower()))
        return {k: sum(v)/len(v) for k, v in m.items()}