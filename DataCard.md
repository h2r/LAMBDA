# LAMBDA

A benchmark consisting of language-conditioned long-horizon (room-to-room and floor-to-floor navigation) MoMa tasks that are paired with human-collected expert demonstrations from both simulated and real-world environments. There are 521 simulated and 50 real trajectories, totalling to 571 trajectories.

#### Dataset Link
<!-- info: Provide a link to the dataset: -->
<!-- width: half -->
https://www.dropbox.com/scl/fo/c1q9s420pzu1285t1wcud/AGMDPvgD5R1ilUFId0i94KE?rlkey=7lwmxnjagi7k9kgimd4v7fwaq&dl=0

#### Data Card Author(s)
<!-- info: Select **one role per** Data Card Author:

(Usage Note: Select the most appropriate choice to describe the author's role
in creating the Data Card.) -->
<!-- width: half -->
- **Name, Team:** Ahmed Jaafar (Owner)

## Authorship
### Publishers
#### Publishing Organization(s)
<!-- scope: telescope -->
<!-- info: Provide the names of the institution or organization responsible
for publishing the dataset: -->
Brown University, Rutgers University, University of Pennsylvania

#### Industry Type(s)
<!-- scope: periscope -->
<!-- info: Select **all applicable** industry types to which the publishing
organizations belong: -->
<!-- - Corporate - Tech -->
<!-- - Corporate - Non-Tech (please specify) -->
- Academic - Tech
<!-- - Academic - Non-Tech (please specify) -->
<!-- - Not-for-profit - Tech
- Not-for-profit - Non-Tech (please specify)
- Individual (please specify)
- Others (please specify) -->

#### Contact Detail(s)
<!-- scope: microscope -->
<!-- info: Provide publisher contact details: -->
- **Publishing POC:** Ahmed Jaafar
- **Affiliation:** Brown University
- **Contact:** ahmed_jaafar@brown.edu
- **Website:** https://lambdabenchmark.github.io/

#### Author(s)
<!-- scope: microscope -->
<!-- info: Provide the details of all authors associated with the dataset:

(Usage Note: Provide the affiliation and year if different from publishing
institutions or multiple affiliations.) -->
- Ahmed Jaafar, Brown University
- Shreyas Sundara Raman, Brown University
- Yichen Wei, Brown University
- Sudarshan Harithas, Brown University
- Sofia Juliani, Rutgers University
- Anneke Wernerfelt, University of Pennsylvania
- Benedict Quartey, Brown University
- Ifrah Idrees, Brown University
- Jason Xinyu Liu, Brown University
- Stefanie Tellex, Associate Professor, Brown University

### Funding Sources

#### Institution(s)
<!-- scope: telescope -->
<!-- info: Provide the names of the funding institution(s): -->
- Office of Naval Research (ONR)
- National Science Foundation (NSF)
- Amazon Robotics

#### Funding or Grant Summary(ies)
<!-- scope: periscope
<!-- width: full -->
<!-- info: Provide a short summary of programs or projects that may have funded
the creation, collection, or curation of the dataset.

Use additional notes to capture any other relevant information or
considerations. -->
 <!-- *For example, Institution 1 and institution 2 jointly funded this dataset as a
part of the XYZ data program, funded by XYZ grant awarded by institution 3 for
the years YYYY-YYYY.* -->

This work is supported by ONR under grant award numbers N00014-22-1-2592 and N00014-23-1-2794, NSF under grant award number CNS-2150184, and with support from Amazon Robotics.


## Dataset Overview
#### Data Subject(s)
<!-- scope: telescope -->
<!-- info: Select ***all applicable**** subjects contained the dataset: -->
- Data about places and objects
- Synthetically generated data
- Data about systems or products and their behaviors
- Others (Language data provided by humans, robot movement and visual data)

#### Dataset Snapshot
<!-- scope: periscope -->
<!-- info: Provide a snapshot of the dataset:<br><br>(Use the additional notes
to include relevant information, considerations, and links to table(s) with
more detailed breakdowns.) -->
Category | Data
--- | ---
Size of Dataset | 288400 MB
Number of Instances | 571
Human Labels | 571
Capabilities | 4
Avg. Trajectory Length | 496 (sim)
Number of environments | 8
Number of rooms | 30
Number of actions | 12
Number of robots | 2


**Above:** The numbers are combining both the simulated and real datasets. "Capabilities" refers to the high-level aspects/modalities this dataset covers: Natural language, Navigation, Manipulation, and Perception. "Human Labels" refers to the natural language commands of robot tasks provided by humans. "Number of actions" refers to the high-level discrete actions in only simulation.

**Additional Notes:** The robots used are mobile manipulators. The simulation robot is from ManipulaTHOR and the real robot is quadruped with an arm, a Boston Dynamics Spot.


#### Content Description
<!-- scope: microscope -->
<!-- info: Provide a short description of the content in a data point: -->
Every data point in simulation (trajectory time step) contains these important aspects: natural language command, egocentric RGB-D, instance segmentations, bounding boxes, robot body pose, robot end-effector pose, and grasped object poses.

Every data point in real (trajectory time step) contains on a high-level: natural language command, egocentric RGB-D, egocentric RGB-D, gripper RGB-D, gripper instance segmentations, robot body pose, robot arm pose, feet positions, joint angles, robot body velocity, robot arm velocity, gripper open percentage, object held boolean.

#### Descriptive Statistics
<!-- width: full -->
<!-- info: Provide basic descriptive statistics for each field.

Use additional notes to capture any other relevant information or
considerations.

Usage Note: Some statistics will be relevant for numeric data, for not for
strings. -->

Statistic | Simulation Trajectories | Real Trajectories 
--- | --- | --- 
count | 521 | 50
mean |496| 323
std | - | 187
min | - | 123
max | - | 733

**Above:** The mean, std, min, and max of the trajectories refers to their lengths.

<!-- **Additional Notes:** Add here. -->

### Sensitivity of Data
#### Sensitivity Type(s)
<!-- scope: telescope -->
<!-- info: Select ***all applicable*** data types present in the dataset: -->
- User Content
- Anonymous Data
- Others (Robot movement and visual data)


#### Risk Type(s)
<!-- scope: telescope -->
<!-- info: Select **all applicable** risk types presenting from the
dataset: -->
- No Known Risks


### Dataset Version and Maintenance
#### Maintenance Status
<!-- scope: telescope -->
<!-- info: Select **one:** -->

**Actively Maintained** - No new versions will be made
available, but this dataset will
be actively maintained,
including but not limited to
updates to the data.


#### Version Details
<!-- scope: periscope -->
<!-- info: Provide details about **this** version of the dataset: -->
**Current Version:** 2.0

**Last Updated:** 03/2025

**Release Date:** 06/2024

#### Maintenance Plan
<!-- scope: microscope -->
<!-- info: Summarize the maintenance plan for the dataset:

Use additional notes to capture any other relevant information or
considerations. -->
Ahmed Jaafar will be maintaining this dataset and resolving dataset issues brought up by the community.

<!-- **Versioning:** Summarize here. Include information about criteria for
versioning the dataset.

**Updates:** Summarize here. Include information about criteria for refreshing
or updating the dataset.

**Errors:** Summarize here. Include information about criteria for refreshing
or updating the dataset.

**Feedback:** Summarize here. Include information about criteria for refreshing
or updating the dataset.

**Additional Notes:** Add here -->

<!-- #### Next Planned Update(s)
<!-- scope: periscope -->
<!-- info: Provide details about the next planned update: -->
<!-- **Version affected:** 1.0

**Next data update:** MM/YYYY

**Next version:** 1.1

**Next version update:** MM/YYYY -->

<!-- #### Expected Change(s)
<!-- scope: microscope -->
<!-- info: Summarize the updates to the dataset and/or data that are expected
on the next update.

Use additional notes to capture any other relevant information or
considerations. -->
<!-- **Updates to Data:** Summarize here. Include links, charts, and visualizations
as appropriate.

**Updates to Dataset:** Summarize here. Include links, charts, and
visualizations as appropriate.

**Additional Notes:** Add here -->

## Example of Data Points
#### Primary Data Modality
<!-- scope: telescope -->
<!-- info: Select **one**: -->
- Multimodel (Natural Language, Vision, Navigation, Manipulation)

<!-- #### Sampling of Data Points
<!-- scope: periscope -->
<!-- info: Provide link(s) to data points or exploratory demos: -->
<!-- - Demo Link
- Typical Data Point Link
- Outlier Data Point Link
- Other Data Point Link
- Other Data Point Link -->

#### Data Fields
<!-- scope: microscope -->
<!-- info: List the fields in data points and their descriptions.

(Usage Note: Describe each field in a data point. Optionally use this to show
the example.) -->

Simulation | Value | Description
--- | --- | ---
Natural Language | "Go pick up the apple and put it on the couch." | The command the human tells the robot for completing a certain task
Scene | "FloorPlan_Train8_1" | The simulation environment in AI2THOR
Sim time | 0.19645 | The simulation time
Wall clock time |  14:49:37 | The real-world time
Body state | [4.0, 6.2, 7.5 , 226] | The global state of the robot, [x, y, z, yaw]
End-effector state | [2.59, 0.89, -4.17, -1.94, -1.27, 1.94] | The global state of the robot's end-effector, [x, y, z, roll, pitch, yaw]
Hand sphere radius | 0.059 | The radius of the hand grasp field
Held objects | [Apple] | A list of objects currently held by the robot
Held object state | [4.4, 2.3, 5.1] | The global state of the currently held objects, [x, y, z]
Bounding boxes  |  {"keys": [Apple], "values":[418, 42, 23, 321]}  | The objects detected with bounding boxes and the coordinates of those boxes
RGB | `./rgb_0.npy` | The path to the RGB npy egocentric image of the time step
Depth | `./depth_0.npy` | The path to the depth npy egocentric image of the time step
Instance segmentations | `./inst_seg_0.npy` | The path to the instance segmentations npy egocentric image of the time step


Real-world | Value | Description
--- | --- | ---
Natural Language | "Go pick up the apple and put it on the couch." | The command the human tells the robot for completing a certain task
Scene | "FloorPlan_Train8_1" | The simulation environment in AI2THOR
Wall clock time |  14:49:37 | The real-world time
Body state | [4.0, 6.2, 7.5] | The global euclidean state of the robot, [x, y, z]
Body state quaternion | [0.04, 0, 0, 0.99] | The global quaternion state of the robot body, [w, x, y, z]
Body orientation  | [0, 0.17, 3.05]  | The global rotation of the robot body, [roll, pitch, yaw]
Body linear velocity | [0, 0.5, 0.1] | The linear velocity of the robot body, [x, y, z]
Body angular velocity | [0, 0.5, 0.1] | The angular velocity of the robot body, [x, y, z]
Arm state | [0.5, 0, 0.26] | The robot arm state relative to the body, [x, y, z]
Arm quaternion state | [0.99, 0, 0.7, 0.008]  | The quaternion robot arm state relative to the body, [w, x, y, z]
Arm state global | [1.9, 0.5, 0] | The global robot arm state, [x, y, z]
Arm quaternion state global | [0.04, 0, 0, 0.99]  | The global quaternion robot arm state, [w, x, y, z]
Arm linear velocity | [0.2, 0.04, 0] | The linear velocity of the robot arm, [x, y, z]
Arm angular velocity | [0.1, 0.4, 0.008] | The angular velocity of the robot arm, [x, y, z]
Arm stowed  |  1  | Boolean of if the arm is stowed or not
Gripper open |  0.512  | The percentage of how open the gripper is
Object held  |  1  | Boolean of if an object is currently held by the gripper
Feet state |  [0.32, 0.17, 0], ... | The state of the four quadruped feet relative to the body, [x, y, z]
Feet state global |  [-0.21, 0.05, 0], ...  |  The global state of the four quadruped feet
Joint angles |   {fl.hx: -0.05, fl.hy: 0.79, fl.kn: -1.57, ...}  |  The angles of all the quadruped's joints
Joint velocities  |   {fl.hx: 0.004, fl.hy: 0.01, fl.kn: 0.57, ...}   | The velocities of all the quadruped's joints
Left RGB | `./left_fisheye_image_0.npy` | The path of the left eye RGB egocentric image, which captures the right side of the view
Right RGB | `./right_fisheye_image_0.npy` | The path of the right eye RGB egocentric image, which captures the left side of the view
Left Depth |  `./left_fisheye_depth_0.npy` | The path of the left eye depth egocentric image, which captures the right side of the view
Right Depth | `./right_fisheye_depth_0.npy` | The path of the right eye depth egocentric image, which captures the left side of the view
Left instance segmentations | `./left_fisheye_image_instance_seg_0.npy` | The path of the left eye instance segmentations egocentric image, which captures the right side of the view
Right instance segmentations | `./right_fisheye_image_instance_seg_0.npy` | The path of the right eye instance segmentations egocentric image, which captures the left side of the view
Gripper RGB | `./gripper_image_0.npy`  | The path of the gripper RGB image
Gripper depth | `./gripper_depth_0.npy`  | The path of the gripper depth image
Gripper instance segmentations | `./gripper_image_instance_seg_0.npy`  | The path of the gripper instance segmentations image

#### Typical Data Point
<!-- width: half -->
<!-- info: Provide an example of a typical data point and describe what makes
it typical.

**Use additional notes to capture any other relevant information or
considerations.** -->

Simulation:
```
{
    "nl_command": "Go to the table and pick up the salt and place it in the white bin in the living room.",
    "scene": "FloorPlan_Train8_1",
    "steps": [
        {
            "sim_time": 0.1852477639913559,
            "wall-clock_time": "15:10:47.900",
            "action": "Initialize",
            "state_body": [3.0, 0.9009992480278015, -4.5, 269.9995422363281],
            "state_ee": [2.5999975204467773, 0.8979992270469666, -4.171003341674805, -1.9440563492718068e-07, -1.2731799533306385, 1.9440386333307377e-07],
            "hand_sphere_radius": 0.05999999865889549
            "held_objs": [],
            "held_objs_state": {},
            "inst_det2D": {
                "keys": [
                    "Wall_4|0.98|1.298|-2.63",
                    "RemoteControl|+01.15|+00.48|-04.24",
                ],
                "values": [
                    [418, 43, 1139, 220], [315, 0, 417, 113], ...
                ]
            },
            "rgb": "./rgb_0.npy",
            "depth": "./depth_0.npy",
            "inst_seg": "./inst_seg_0.npy",
        }
    ]
}
```

Real-world:

```
{
  "language_command": "Go pick up Hershey's syrup in the room with the big window and bring it to the room with the other Spot.",
  "scene_name": "",
  "wall_clock_time": "12:50:10.923",
  "left_fisheye_rgb": "./Trajectories/trajectories/data_3/folder_0.zip/left_fisheye_image_0.npy",
  "left_fisheye_depth": "./Trajectories/trajectories/data_3/folder_0.zip/left_fisheye_depth_0.npy",
  "right_fisheye_rgb": "./Trajectories/trajectories/data_3/folder_0.zip/right_fisheye_image_0.npy",
  "right_fisheye_depth": "./Trajectories/trajectories/data_3/folder_0.zip/right_fisheye_depth_0.npy",
  "gripper_rgb": "./Trajectories/trajectories/data_3/folder_0.zip/gripper_image_0.npy",
  "gripper_depth": "./Trajectories/trajectories/data_3/folder_0.zip/gripper_depth_0.npy",
  "left_fisheye_instance_seg": "./Trajectories/trajectories/data_3/folder_0.zip/left_fisheye_image_instance_seg_0.npy",
  "right_fisheye_instance_seg": "./Trajectories/trajectories/data_3/folder_0.zip/right_fisheye_image_instance_seg_0.npy",
  "gripper_fisheye_instance_seg": "./Trajectories/trajectories/data_3/folder_0.zip/gripper_image_instance_seg_0.npy",
  "body_state": {"x": 1.7732375781707208, "y": -0.2649551302417769, "z": 0.04729541059536978},
  "body_quaternion": {"w": 0.11121513326494507, "x": 0.00003060940357089109, "y": 0.0006936040684443222, "z": 0.9937961119411372},
  "body_orientation": {"r": 0.0017760928400286857, "p": 0.016947586302323542, "y": 2.919693676695565},
  "body_linear_velocity": {"x": 0.0007985030885781894, "y": 0.0007107887103978708, "z": -0.00001997174236456424},
  "body_angular_velocity": {"x": -0.002894917543479851, "y": -0.0017834609980581554, "z": 0.00032649917985633773},
  "arm_state_rel_body": {"x": 0.5536401271820068, "y": 0.0001991107128560543, "z": 0.2607555091381073},
  "arm_quaternion_rel_body": {"w": 0.9999642968177795, "x": 0.00019104218517895788, "y": 0.008427758701145649, "z": 0.008427758701145649},
  "arm_orientation_rel_body": {"x": 0.0003903917486135314, "y": 0.016855526363847233, "z":0.0009807885066525242},
  "arm_state_global": {"x": 1.233305266138133, "y": 0.0001991107128560543, "z": 0.2607555091381073},
  "arm_quaternion_global": {"w": 0.11071797661404018, "x": -0.0083232786094425, "y": 0.0018207155823512953, "z": 0.9938152930378756},
  "arm_orientation_global": {"x": 0.0017760928400286857, "y": 0.016947586302323542, "z": 2.919693676695565},
  "arm_linear_velocity": {"x": -0.00015927483240388228, "y": 0.00006229256340773636, "z": -0.003934306244239418},
  "arm_angular_velocity": {"x": 0.02912604479413378, "y": -0.012041083915871545, "z": 0.009199674753842119},
  "arm_stowed": 1,
  "gripper_open_percentage": 0.521618127822876,
  "object_held": 0,
  "feet_state_rel_body": [
    {"x": 0.32068437337875366, "y": 0.17303785681724548, "z": -0.5148577690124512},
    {"x": 0.32222312688827515, "y": -0.17367061972618103, "z": -0.5163648128509521},
    ...
  ],
  "feet_state_global": [
    {"x": -0.35111223090819643, "y": -0.0985760241189894, "z": -0.5146475087953596},
    {"x": -0.27597323368156573, "y": 0.239893453842677, "z": -0.5166350285289446},
    ...
  ],
  "all_joint_angles": {"fl.hx": 0.013755097053945065, "fl.hy": 0.7961212992668152, "fl.kn": -1.5724135637283325, ...},
  "all_joint_velocities": {"fl.hx": -0.007001522462815046, "fl.hy": 0.0006701984675601125, "fl.kn": 0.00015050712681841105, ...}
}
```



## Motivations & Intentions
### Motivations
#### Purpose(s)
<!-- scope: telescope -->
<!-- info: Select **one**: -->
- Research

#### Domain(s) of Application
<!-- scope: periscope -->
<!-- info: Provide a list of key domains of application that the dataset has
been designed for:<br><br>(Usage Note: Use comma-separated keywords.) -->

`Robotics`, `Imitation Learning`, `Behavior Cloning`, `Reinfocement Learning`, `Machine Learning`

#### Motivating Factor(s)
<!-- scope: microscope -->
<!-- info: List the primary motivations for creating or curating this dataset:

(Usage Note: use this to describe the problem space and corresponding
motivations for the dataset.) -->

<!-- There have been recent advances in robotic mobile manipulation, however the field as a whole is still lagging behind. We feel one reason behind this is a lack of useful and difficult benchmarks for mobile manipulation models. In particular, there were no benchmarks that have data for long-horizon room-to-room pick-and-place tasks comprised of natural langauge, navigation, manipulation, and perception in both simulation and the real-world, including a quadruped. -->

### Intended Use
#### Dataset Use(s)
<!-- scope: telescope -->
<!-- info: Select **one**: -->
- Safe for research use

#### Suitable Use Case(s)
<!-- scope: periscope -->
<!-- info: Summarize known suitable and intended use cases of this dataset.

Use additional notes to capture any specific patterns that readers should
look out for, or other relevant information or considerations. -->
**Suitable Use Case:** Training and testing behavior cloning models for data efficiency.

**Suitable Use Case:** Learning reward functions via inverse reinforcement learning.

**Suitable Use Case:** Robot skill learning.

**Suitable Use Case:** Providing in-context examples for robot planning.

<!-- **Additional Notes:** Add here -->


#### Research and Problem Space(s)
<!-- scope: periscope -->
<!-- info: Provide a description of the specific problem space that this
dataset intends to address. -->
<!-- This dataset intendes to serve as a benchmark addressing the gap of the integration of natural language, navigation, manipulation, and perception for pick-and-place mobile manipulation tasks that span room-to-room and floor-to-floor in both simulated and real environments. Mobile manipulation is lagging behind overall, and we believe one of the reasons behind that is a lack of difficult comprehensive benchmarks that models in developement can be tested against. LaNMP is here to fill this gap. -->

#### Citation Guidelines
<!-- scope: microscope -->
<!-- info: Provide guidelines and steps for citing this dataset in research
and/or production.

Use additional notes to capture any specific patterns that readers should look
out for, or other relevant information or considerations. -->
**Guidelines & Steps:** Referncing the BiBTeX below.

**BiBTeX:**
```
 @misc{lambdabenchmark,
      title={{\lambda}: A Benchmark for Data-Efficiency in Long-Horizon Indoor Mobile Manipulation Robotics}, 
      author={Ahmed Jaafar and Shreyas Sundara Raman and Yichen Wei and Sofia Juliani and Anneke Wernerfelt and Benedict Quartey and Ifrah Idrees and Jason Xinyu Liu and Stefanie Tellex},
      year={2025},
      eprint={2412.05313},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2412.05313}, 
    }
```

## Access
### Access
#### Access Type
<!-- scope: telescope -->
<!-- info: Select **one**: -->
- External - Open Access

#### Documentation Link(s)
<!-- scope: periscope -->
<!-- info: Provide links that describe documentation to access this
dataset: -->
- Dataset Website URL: https://www.dropbox.com/scl/fo/c1q9s420pzu1285t1wcud/AGMDPvgD5R1ilUFId0i94KE?rlkey=7lwmxnjagi7k9kgimd4v7fwaq&dl=0
- GitHub URL: https://github.com/h2r/LAMBDA/


## Provenance
### Collection
#### Method(s) Used
<!-- scope: telescope -->
<!-- info: Select **all applicable** methods used to collect data: -->

- Crowdsourced - Paid
- Crowdsourced - Volunteer
- Survey, forms, or polls
- Others (Keyboard teleoperated, tablet-controller teleoperated)

#### Methodology Detail(s)
<!-- scope: periscope -->
<!-- info: Provide a description of each collection method used.

Use additional notes to capture any other relevant information or
considerations.

(Usage Note: Duplicate and complete the following for collection method
type.) -->
**Collection Type**

**Source:** Prolific.

**Platform:** [Prolific](www.prolific.com), A crowdsourcing platform for researchers to collect data.

**Is this source considered sensitive or high-risk?** No

**Dates of Collection:** [03 2024 - 04 2024]

**Primary modality of collection data:**

- Text Data

**Update Frequency for collected data:**

- Static

**Additional Notes:** Used to collect the natural language commands. Crowdsourced humans explore the simulated environements and come up with commands for tasks the robot can do in those environements.


**Collection Type**

**Source:** Human teleoperation

**Platform:** [AI2THOR](https://ai2thor.allenai.org/) simulator

**Is this source considered sensitive or high-risk?** No

**Dates of Collection:** [09 2024]

**Primary modality of collection data:**

- Multimodal (Navigation, Manipulation, Vision)

**Update Frequency for collected data:**

- Static

**Additional Notes:** Humans teleoperate a simulated robot via keyboard to collect the robot trajectory data.


**Collection Type**

**Source:** Human speech

**Platform:** N/A

**Is this source considered sensitive or high-risk?** No

**Dates of Collection:** [05 2024]

**Primary modality of collection data:**

- Text Data

**Update Frequency for collected data:**

- Static

**Additional Notes:** Used to collect the natural language commands. Humans explore the real-world environements and come up with commands for tasks the robot can do in those environements.


**Collection Type**

**Source:** Human teleoperation

**Platform:** [Boston Dynamics Spot](https://bostondynamics.com/products/spot/)

**Is this source considered sensitive or high-risk?** No

**Dates of Collection:** [05 2024]

**Primary modality of collection data:**

- Multimodal (Navigation, Manipulation, Vision)

**Update Frequency for collected data:**

- Static

**Additional Notes:** Human teleoperates a real quadruped robot via a tablet/joystick controller to collect the robot trajectory data.

#### Collection Cadence
<!-- scope: telescope -->
<!-- info: Select **all applicable**: -->
**Static:** Data was collected once from single or multiple sources.

#### Data Processing
<!-- scope: microscope -->
<!-- info: Summarize how data from different sources or methods aggregated,
processed, or connected.

Use additional notes to capture any other
relevant information or considerations.

(Usage Note: Duplicate and complete the following for each source OR
collection method.) -->
**Collection Method or Source**

**Description:** Natural language commands

**Methods employed:** Utilized other humans to manually correct grammatical mistakes in the given textual natural language commands. The humans deleted the commands that were not possible for the robot to execute or did not match the desired research goal.

**Tools or libraries:** N/A

**Collection Method or Source**

**Description:** Robot trajectories

**Methods employed:** Utilized other humans to manually delete incomplete collected trajectories.

**Tools or libraries:** N/A

### Collection Criteria
#### Data Selection
<!-- scope: telescope -->
<!-- info: Summarize the data selection criteria.

Use additional notes to capture any other relevant information or
considerations. -->
- **Natural language commands:** The criteria for selction included commands that mention a pick-and-place task where the robot picks up an object and places it somewhere else, and having the robot go from room-to-room.
- **Trajectories:** The criteria for selction included trajectories that execute the commands in the most efficient manner, ones that minimize robot lag, and ones that don't collide objects in the environment.

### Relationship to Source

#### Benefit and Value(s)
<!-- scope: periscope -->
<!-- info: Summarize the benefits of the resulting dataset to its consumers,
compared to the upstream source(s).

Use additional notes to capture any other relevant information or
considerations.

(Usage Note: Duplicate and complete the following for each source type.) -->
- Combines natural language, navigation, manipulation, and perception robot data
- Mobile manipulation pick-and-place tasks that are room-to-room and some are cross-floor making them long-horizon
- Utilizing a quadruped which can handle terrains that other robots can't, such as stairs, enabling cross-floor tasks
- Diverse environements and objects


#### Limitation(s) and Trade-Off(s)
<!-- scope: microscope -->
<!-- info: What are the limitations of the resulting dataset to its consumers,
compared to the upstream source(s)?

Break down by source type.<br><br>(Usage Note: Duplicate and complete the
following for each source type.) -->
- Only pick-and-place tasks
- No ground-truth goal position of the target object
- Size

<!-- ### Version and Maintenance
<!-- info: Fill this next row if this is not the first version of the dataset,
and there is no data card available for the first version -->
<!-- #### First Version -->
<!-- scope: periscope -->
<!-- info: Provide a **basic description of the first version** of this
dataset. -->
<!-- - **Release date:** MM/YYYY
- **Link to dataset:** [Dataset Name + Version]
- **Status:** [Select one: Actively Maintained/Limited Maintenance/Deprecated]
- **Size of Dataset:** 123 MB
- **Number of Instances:** 123456 -->
<!-- 
#### Note(s) and Caveat(s)
<!-- scope: microscope -->
<!-- info: Summarize the caveats or nuances of the first version of this
dataset that may affect the use of the current version.

Use additional notes to capture any other relevant information or
considerations. -->
<!-- Summarize here. Include links where available.

**Additional Notes:** Add here -->

<!-- #### Cadence
<!-- scope: telescope -->
<!-- info: Select **one**: -->
<!-- - Yearly
- Quarterly
- Monthly
- Biweekly
- Weekly
- Daily
- Hourly
- Static
- Others (please specify) -->

<!-- #### Last and Next Update(s) -->
<!-- scope: periscope -->
<!-- info: Please describe the update schedule: -->
<!-- - **Date of last update:** DD/MM/YYYY
- **Total data points affected:** 12345
- **Data points updated:** 12345
- **Data points added:** 12345
- **Data points removed:** 12345
- **Date of next update:** DD/MM/YYYY -->

<!-- #### Changes on Update(s)
<!-- scope: microscope -->
<!-- info: Summarize the changes that occur when the dataset is refreshed.

Use additional notes to capture any other relevant information or
considerations.

(Usage Note: Duplicate and complete the following for each source type.) -->
<!-- - **Source Type:** Summarize here. Include links where available.
- **Source Type:** Summarize here. Include links where available.
- **Source Type:** Summarize here. Include links where available.

**Additional Notes:** Add here -->

## Human and Other Sensitive Attributes
#### Sensitive Human Attribute(s)
<!-- scope: telescope -->
<!-- info: Select **all attributes** that are represented (directly or
indirectly) in the dataset. -->
- Language

#### Intentionality
<!-- scope: periscope -->
<!-- info: List fields in the dataset that contain human attributes, and
specify if their collection was intentional or unintentional.

Use additional notes to capture any other relevant information or
considerations. -->
**Intentionally Collected Attributes**

Human attributes were labeled or collected as a part of the dataset creation
process.

Field Name | Description
--- | ---
nl_command | Natural language commands given by humans telling the robot what task to do in the simulator
language_command | Natural language commands given by humans telling the robot what task to do in the real-world


**Unintentionally Collected Attributes**

Human attributes were not explicitly collected as a part of the dataset
creation process but can be inferred using additional methods.

N/A

#### Rationale
<!-- scope: microscope -->
<!-- info: Describe the motivation, rationale, considerations or approaches
that caused this dataset to include the indicated human attributes.

Summarize why or how this might affect the use of the dataset. -->
We wanted to capture a natural distribution of commands that humans would tell a househould robots to complete long-horizon mobile manipulation tasks. Rather than automatically generating the commands using tools such as LLMs, we wanted to capture what humans really want done in households by assitant robots, so we used humans to provide the commands. Since the ultimate goal is to one day have assistive robots in the home and workplace, capturing the commands that humans would eventually tell them now is crucial for research and development to get to that goal.

#### Source(s)
<!-- scope: periscope -->
<!-- info: List the sources of the human attributes.

Use additional notes to capture any other relevant information or
considerations. -->
- **Human Attribute:** Prolific.com
- **Human Attribute:** In-person humans

## Extended Use
### Use with Other Data
#### Safety Level
<!-- scope: telescope -->
<!-- info: Select **one**: -->
- Safe to use with other data

#### Best Practices
<!-- scope: microscope -->
<!-- info: Summarize best practices for using this dataset with other datasets
or data types.

Use additional notes to capture any other relevant information or
considerations. -->
- Make sure the datasets are both in the same format
- Do not mix at the time step level, only at the trajectory level, e.g. Other dataset trajectory Y can come after LaNMP trajectory X, but X and Y's time steps should not be mixed

### Forking & Sampling
#### Safety Level
<!-- scope: telescope -->
<!-- info: Select **one**: -->
- Safe to form and/or sample

#### Acceptable Sampling Method(s)
<!-- scope: periscope -->
<!-- info: Select **all applicable** acceptable methods to sample this
dataset: -->
- Cluster Sampling
- Haphazard Sampling
- Multi-stage sampling
- Random Sampling
- Stratified Sampling
- Systematic Sampling
- Weighted Sampling

#### Best Practice(s)
<!-- scope: microscope -->
<!-- info: Summarize the best practices for forking or sampling this dataset.

Use additional notes to capture any other relevant information or
considerations. -->
Do not sample at the time step level, only at the trajectory level, e.g. sample trajectories 4-15 but not the timesteps of those trajectories.

### Use in ML or AI Systems
#### Dataset Use(s)
<!-- scope: telescope -->
<!-- info: Select **all applicable** -->
- Training
- Testing
- Validation
- Fine Tuning

#### Notable Feature(s)
<!-- scope: periscope -->
<!-- info: Describe any notable feature distributions or relationships between
individual instances made explicit.

Include links to servers where readers can explore the data on their own. -->

**Exploration Demo:** [Google Colab notebook](https://colab.research.google.com/drive/18fTkjqcvlyOkCkbou6LK2RG2XKsPT__K?usp=sharing)


#### Distribution(s)
<!-- scope: periscope -->
<!-- info: Describe the recommended splits and corresponding criteria.

Use additional notes to capture any other
relevant information or considerations. -->

<!-- Set | Number of data points
--- | ---
Train | 446
Test | 78

**Above:** We don't hyperparameter tune so we only use train and test splits. 85% and 15% respectively. This is only for the simulation data.

**Additional Notes:** This split was only used during the task generalization experiment. More details in the paper. -->

#### Split Statistics
<!-- scope: periscope -->
<!-- width: full -->
<!-- info: Provide the sizes of each split. As appropriate, provide any
descriptive statistics for features. -->

<!-- Statistic | Train | Test 
--- | --- | --- 
Count | 446 | 78 


**Above:** We don't hyperparameter tune so we only use train and test splits. 85% and 15% respectively. This is only for the simulation data. -->


## Transformations
<!-- info: Fill this section if any transformations were applied in the
creation of your dataset. -->
### Synopsis
#### Transformation(s) Applied
<!-- scope: telescope -->
<!-- info: Select **all applicable** transformations
that were applied to the dataset. -->
- Other (Fixing natural langauge command grammatical mistakes)

#### Field(s) Transformed
<!-- scope: periscope -->
<!-- info: Provide the fields in the dataset that
were transformed.

Use additional notes to capture any
other relevant information or
considerations.

(Usage Note: Duplicate and complete
the following for each transformation
type applied. Include the data types to
which fields were transformed.) -->
**Transformation Type**

Field Name | Description
--- | ---
nl_command | Natural language commands given by humans telling the robot what task to do in the simulator
language_command | Natural language commands given by humans telling the robot what task to do in the real-world

**Additional Notes:** Fixing grammatical mistakes of the commands or deleting trajectories where the commands are incomplete.

#### Library(ies) and Method(s) Used
<!-- scope: microscope -->
<!-- info: Provide a description of the methods
used to transform or process the
dataset.

Use additional notes to capture any
other relevant information or
considerations.

(Usage Note: Duplicate and complete
the following for each transformation
type applied.) -->
**Transformation Type**

**Method:** Manually fixing grammatically incorrect natural language commands and injecting them into their respective trajectories to replace the already saved wrong commands. Also deleting trajectories that have incomplete commands e.g. "Pick up the blue"

**Transformation Results:** Trajectories with the fixed commands, and less trajectories overall due to the deletion of the ones that had incompelete commands.

## Annotations & Labeling
<!-- info: Fill this section if any human or algorithmic annotation tasks were
performed in the creation of your dataset. -->
#### Annotation Workforce Type
<!-- scope: telescope -->
<!-- info: Select **all applicable** annotation
workforce types or methods used
to annotate the dataset: -->
- Human Annotations (Expert)
- Human Annotations (Non-Expert)
- Human Annotations (Employees)
- Human Annotations (Crowdsourcing)

#### Annotation Characteristic(s)
<!-- scope: periscope -->
<!-- info: Describe relevant characteristics of annotations
as indicated. For quality metrics, consider
including accuracy, consensus accuracy, IRR,
XRR at the appropriate granularity (e.g. across
dataset, by annotator, by annotation, etc.).

Use additional notes to capture any other
relevant information or considerations.

(Usage Note: Duplicate and complete the
following for each annotation type.) -->
**Expert** | **Number**
--- | ---
Number of unique annotations | 50
Total number of annotations | 50
Average annotations per example | 1
Number of annotators | 1
Number of annotators per example | 1
<!-- [Quality metric per granuality] | 123456789
[Quality metric per granuality] | 123456789
[Quality metric per granuality] | 123456789 -->

**Above:** The real-world robot trajectory execution (teleoperation) data collection done by one of the authors.

**Non-Expert** | **Number**
--- | ---
Number of unique annotations | 50
Total number of annotations | 50
Average annotations per example | 1
Number of annotators | 7
Number of annotators per example | 1

**Above:** Humans that gave natural language commands of tasks for the real-world robot to execute.

**Employees** | **Number**
--- | ---
Number of unique annotations | 524
Total number of annotations | 524
Average annotations per example | 1
Number of annotators | 15
Number of annotators per example | 1

**Above:** Humans that exected the trajectories in the simulator.

**Crowdsourcing** | **Number**
--- | ---
Number of unique annotations | 521
Total number of annotations | 521
Average annotations per example | 1
Number of annotators | 41
Number of annotators per example | 1

**Above:**  Humans that gave natural language commands of tasks for the simulated robot to execute.


#### Annotation Description(s)
<!-- scope: microscope -->
<!-- info: Provide descriptions of the annotations
applied to the dataset. Include links
and indicate platforms, tools or libraries
used wherever possible.

Use additional notes to capture any
other relevant information or
considerations.

(Usage Note: Duplicate and complete
the following for each annotation
type.) -->

**Expert**

**Description:** The real-world robot trajectory execution (teleoperation) data collection done by one of the authors.

**Link:** N/A

**Platforms, tools, or libraries:**

- Boston Dynamics Spot

**Non-Expert**

**Description:** Humans that gave natural language commands of tasks for the real-world robot to execute.

**Link:** N/A

**Platforms, tools, or libraries:**

- N/A

**Employees**

**Description:** Humans that exected the trajectories in the simulator.

**Link:** https://ai2thor.allenai.org/

**Platforms, tools, or libraries:**

- AI2THOR

**Crowdsourcing**

**Description:** Humans that gave natural language commands of tasks for the simulated robot to execute.

**Link:** https://www.prolific.com/

**Platforms, tools, or libraries:**

- Prolific

### Human Annotators
<!-- info: Fill this section if human annotators were used. -->
#### Annotator Description(s)
<!-- scope: periscope -->
<!-- info: Provide a brief description for each annotator
pool performing the human annotation task.

Use additional notes to capture any other
relevant information or considerations.

(Usage Note: Duplicate and complete the
following for each annotation type.) -->
**Expert Real-Robot Trajectory Collection**

**Task type:**  The real-world robot trajectory execution (teleoperation) data collection done by one of the authors

**Number of unique annotators:** 1

**Expertise of annotators:** Expert

**Description of annotators:** An author.

**Language distribution of annotators:** English

**Geographic distribution of annotators:** United States

**Annotation platforms:** Boston Dyanmics Spot


**Non-Expert Real-Robot Command Collection**

**Task type:**  Humans that gave natural language commands of tasks for the real-world robot to execute

**Number of unique annotators:** 7

**Expertise of annotators:** Non-Expert

**Description of annotators:** Students

**Language distribution of annotators:** English

**Geographic distribution of annotators:** United States

**Annotation platforms:** N/A


**Employed Simulator Command Collection**

**Task type:**  Humans that exected the trajectories in the simulator

**Number of unique annotators:** 7

**Expertise of annotators:** Non-Expert

**Description of annotators:** General adults

**Language distribution of annotators:** English

**Geographic distribution of annotators:** United States and United Kingdom

**Annotation platforms:** Prolific.com


#### Language(s)
<!-- scope: telescope -->
<!-- info: Provide annotator distributions for
each annotation type.

Use additional notes to capture any
other relevant information or
considerations.

(Usage Note: Duplicate and
complete the following for each
annotation type.) -->
- English [100%]

**Above:** All the natural language commands.

<!-- ## Validation Types -->
<!-- info: Fill this section if the data in the dataset was validated during
or after the creation of your dataset. -->
<!-- #### Method(s) -->
<!-- scope: telescope -->
<!-- info: Select **all applicable**: -->
<!-- - Data Type Validation
- Range and Constraint Validation
- Code/cross-reference Validation
- Structured Validation
- Consistency Validation
- Not Validated
- Others (Please Specify) -->

<!-- #### Breakdown(s) -->
<!-- scope: periscope -->
<!-- info: Provide a description of the fields and data
points that were validated.

Use additional notes to capture any other
relevant information or considerations.

(Usage Note: Duplicate and complete the
following for each validator type.) -->
<!-- **(Validation Type)**

**Number of Data Points Validated:** 12345

**Fields Validated**
Field | Count (if available)
--- | ---
Field | 123456
Field | 123456
Field | 123456

**Above:** Provide a caption for the above table or visualization. -->

<!-- #### Description(s) -->
<!-- scope: microscope -->
<!-- info: Provide a description of the methods used to
validate the dataset.

Use additional notes to capture any other
relevant information or considerations.

(Usage Note: Duplicate and complete the
following for each validator type.) -->
<!-- **(Validation Type)**

**Method:** Describe the validation method here. Include links where
necessary.

**Platforms, tools, or libraries:**

- Platform, tool, or library: Write description here
- Platform, tool, or library: Write description here
- Platform, tool, or library: Write description here

**Validation Results:** Provide results, outcomes, and actions taken because
of the validation. Include visualizations where available.

**Additional Notes:** Add here -->

<!-- ### Description of Human Validators -->
<!-- info: Fill this section if the dataset was validated using human
validators -->
<!-- #### Characteristic(s) -->
<!-- scope: periscope -->
<!-- info: Provide characteristics of the validator
pool(s). Use additional notes to capture any
other relevant information or considerations. -->
<!-- **(Validation Type)**
- Unique validators: 12345
- Number of examples per validator: 123456
- Average cost/task/validator: $$$
- Training provided: Y/N
- Expertise required: Y/N -->

<!-- #### Description(s) -->
<!-- scope: microscope -->
<!-- info: Provide a brief description of the validator
pool(s). Use additional notes to capture any
other relevant information or considerations.

(Usage Note: Duplicate and complete the
following for each validator type.) -->
<!-- **(Validation Type)**

**Validator description:** Summarize here. Include links if available.

**Training provided:** Summarize here. Include links if available.

**Validator selection criteria:** Summarize here. Include links if available.

**Training provided:** Summarize here. Include links if available.

**Additional Notes:** Add here -->

## Sampling Methods
<!-- info: Fill out the following block if your dataset employs any sampling
methods. -->
#### Method(s) Used
<!-- scope: telescope -->
<!-- info: Select **all applicable** methods used in the creation of this
dataset: -->
- Unsampled

## Known Applications & Benchmarks
<!-- info: Fill out the following section if your dataset was primarily
created for use in AI or ML system(s) -->
#### ML Application(s)
<!-- scope: telescope -->
<!-- info: Provide a list of key ML tasks
that the dataset has been
used for.

Usage Note: Use comma-separated keywords. -->
*Classification, Regression, Supervised Learning, Imitation Learning*

#### Evaluation Result(s)
<!-- scope: periscope -->
<!-- info: Provide the evaluation results from
models that this dataset has been used
in.

Use additional notes to capture any
other relevant information or
considerations.

(Usage Note: Duplicate and complete the
following for each model.) -->
**RT-1**

**Model Card:** In page 21 of the [paper](https://robotics-transformer.github.io/assets/rt1.pdf).

**Additional Notes:** These results are from the simulation data only.

#### Evaluation Process(es)
<!-- scope: microscope -->
<!-- info: Provide a description of the evaluation process for
the model's overall performance or the
determination of how the dataset contributes to
the model's performance.

Use additional notes to capture any other relevant
information or considerations.

(Usage Note: Duplicate and complete the following
for each model and method used.) -->

**[Metrics used]:**

**Additional Notes:** 

#### Description(s) and Statistic(s)
<!-- scope: periscope -->
<!-- info: Provide a description of the model(s) and
task(s) that this dataset has been used
in.

Use additional notes to capture any
other relevant information or
considerations.

(Usage Note: Duplicate and complete the
following for each model.) -->
**RT-1**

**Model Card:** In page 21 of the [paper](https://robotics-transformer.github.io/assets/rt1.pdf).

**Model Description:** Robotics Transformer 1 (RT-1) is a model designed for generalizing across large-scale, multi-task datasets with real-time inference capabilities. RT-1 leverages a Transformer architecture to process images and natural language instructions to generate discretized actions for mobile manipulation. RT-1 is trained on a diverse dataset of approximately 130K episodes across more
than 700 tasks collected using 13 robots. This enables RT-1 to learn through BC from human
demonstrations annotated with detailed instructions.

- Model Size: 35M (params)


#### Expected Performance and Known Caveats
<!-- scope: microscope -->
<!-- info: Provide a description of the expected performance
and known caveats of the models for this dataset.

Use additional notes to capture any other relevant
information or considerations.

(Usage Note: Duplicate and complete the following
for each model.) -->

**Expected Performance:** 

**Known Caveats:** 