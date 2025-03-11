from ai2thor.controller import Controller
import argparse

# Step 2: Create an ArgumentParser object
parser = argparse.ArgumentParser(description="Run the controller with specified scene")

# Step 3: Add the scene argument
parser.add_argument("--scene", type=str, required=True, help="Specify the scene to load. Either 's1', 's2', 's3', 's4', 's5'")
parser.add_argument("--command", type=str, required=True, help="Specify the natural language command")

# Step 4: Parse the command line arguments
args = parser.parse_args()

scene = ''
if args.scene == 's1':
    scene = 'FloorPlan_Train1_3'
elif args.scene == 's2':
    scene = 'FloorPlan_Train5_1'
elif args.scene == 's3':
    scene = 'FloorPlan_Train7_5'
elif args.scene == 's4':
    scene = 'FloorPlan_Train8_1'
elif args.scene == 's5':
    scene = 'FloorPlan_Train12_3'
else:
    raise ValueError(f"Invalid scene name: {args.scene}. Pick from 's1', 's2', 's3', 's4', 's5'")

if __name__ == "__main__":
    controller = Controller(
        massThreshold = 1,
        agentMode="arm",
        scene = scene,
        snapToGrid=False,
        visibilityDistance=1.5,
        gridSize=0.25,
        renderDepthImage=True,
        renderInstanceSegmentation=True,
        # width= 256,
        # height= 144,
        width= 1280,
        height= 720,
        fieldOfView=60
    )

    while True:
        controller.interact(args.command)#metadata=True, color_frame=True, depth_frame=True, semantic_segmentation_frame=True, instance_segmentation_frame=True)