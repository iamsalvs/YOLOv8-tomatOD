import os
import torch
import torch.nn as nn
import torchvision.models as models
from ultralytics import YOLO

class CustomBackbone(nn.Module):
    """EfficientNet-B0 Backbone for YOLOv8 with Feature Map Extraction and Channel Adjustment"""
    
    def __init__(self, backbone_type='efficientnet', **kwargs):
        super().__init__()
        
        if backbone_type == 'efficientnet':
            self.backbone = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
            self.out_channels = [24, 40, 112]  # EfficientNet feature maps

        # Extract feature layers from EfficientNet
        self.feature_layers = self._get_feature_layers()

        # Channel adapters to match YOLO expectations
        self.channel_adapter = nn.ModuleList([
            nn.Conv2d(24, 256, kernel_size=1),  # Adjust P3: 24 -> 256
            nn.Conv2d(40, 512, kernel_size=1),  # Adjust P4: 40 -> 512
            nn.Conv2d(112, 1024, kernel_size=1) # Adjust P5: 112 -> 1024
        ])

    def _get_feature_layers(self):
        """Extracts feature layers from EfficientNet"""
        return nn.ModuleDict({
            'P3': nn.Sequential(self.backbone.features[:3]),  # P3/8
            'P4': nn.Sequential(self.backbone.features[3:5]), # P4/16
            'P5': nn.Sequential(self.backbone.features[5:])   # P5/32
        })

    def forward(self, x):
        features = []
        for i, (name, layer) in enumerate(self.feature_layers.items()):
            x = layer(x)
            x = self.channel_adapter[i](x)  # Adjust feature map channels
            print(f"✅ Feature Map {name}: {x.shape}")  # Debugging output
            features.append(x)
        return features  # Returns [P3, P4, P5] to YOLO head

def modify_model_for_efficientnet(yolo_model):
    """Replace YOLO's default backbone with EfficientNet-B0"""
    custom_backbone = CustomBackbone(backbone_type='efficientnet')

    # Ensure YOLO model follows expected structure
    if hasattr(yolo_model.model, 'model') and len(yolo_model.model.model) > 0:
        yolo_model.model.model[0] = custom_backbone  # Replace YOLO backbone
    else:
        print("❌ Error: YOLOv8 model structure not recognized!")

    # Debugging: Print model structure
    print("✅ Modified YOLO Model Structure:\n", yolo_model.model)
    
    return yolo_model

def run_yolo_training():
    # SETTINGS: Update dataset path
    MODEL_PATH = "YOLOv8n.pt"
    NAME = 'tomatOD_run_effnet'
    DATA = 'tomatOD_yolo/data.yaml'  
    EPOCHS = 25
    BATCH = 16
    IMGSZ = 300
    LR0 = 0.01
    LRF = 0.0001
    MOMENTUM = 0.9
    OPTIMIZER = 'SGD'
    SAVE_PERIOD = 1
    VAL = True

    # ✅ Load YOLO model
    model = YOLO(MODEL_PATH)

    # ✅ Modify YOLO model to use EfficientNet
    model = modify_model_for_efficientnet(model)

    # Debugging: Check if EfficientNet is being used
    print("🔍 Model after modification:\n", model.model)

    # ✅ Train the model
    train_results = model.train(
        data=DATA,
        epochs=EPOCHS,
        batch=BATCH,
        imgsz=IMGSZ,
        workers=0,
        save_period=SAVE_PERIOD,
        device=0,
        name=NAME,
        optimizer=OPTIMIZER,
        lr0=LR0,
        lrf=LRF,
        momentum=MOMENTUM,
        val=VAL,
        project="runs/detect"
    )

    # ✅ Validate checkpoints
    weights_path = os.path.join('runs', 'detect', NAME, 'weights')
    start_epoch = 1
    end_epoch = EPOCHS
    interval = 1
    imgsz_val = IMGSZ
    batch_val = 1
    save_json = True
    conf = 0.01
    iou = 0.5
    max_det = 50

    def write_results(name, metrics, split="val"):
        output_dir = os.path.join('runs', 'detect', name)
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, f'{split}_ap50.txt'), 'a') as file:
            for ap50 in metrics.box.ap50:
                file.write(str(ap50) + '\n')
            file.write('\n')
            file.write(str(metrics.box.map50))
        with open(os.path.join(output_dir, f'{split}_maps.txt'), 'a') as file:
            for m in metrics.box.maps:
                file.write(str(m) + '\n')
            file.write('\n')
            file.write(str(metrics.box.map))

    # ✅ Loop over epochs and validate only if the checkpoint exists
    for epoch in range(start_epoch, end_epoch, interval):
        weight_file = os.path.join(weights_path, f'epoch{epoch}.pt')
        if not os.path.exists(weight_file):
            print(f"❌ Weight file {weight_file} does not exist, skipping validation for epoch {epoch}.")
            continue
        model = YOLO(weight_file)
        run_label = f"{NAME}_epoch_{epoch}"

        # Validate on validation set
        val_metrics = model.val(
            imgsz=imgsz_val,
            batch=batch_val,
            workers=0,
            save_json=save_json,
            conf=conf,
            iou=iou,
            max_det=max_det,
            name=f"{run_label}_val",
            split="val",
            project="runs/detect"
        )
        write_results(run_label, val_metrics, split="val")

        # Validate on test set
        test_metrics = model.val(
            imgsz=imgsz_val,
            batch=batch_val,
            workers=0,
            save_json=save_json,
            conf=conf,
            iou=iou,
            max_det=max_det,
            name=f"{run_label}_test",
            split="test",
            project="runs/detect"
        )
        write_results(run_label, test_metrics, split="test")

    # ✅ Validate the final model (last.pt) if available
    weight_file = os.path.join(weights_path, 'last.pt')
    if os.path.exists(weight_file):
        model = YOLO(weight_file)
        run_label = f"{NAME}_epoch_{EPOCHS}"

        # Final validation
        val_metrics = model.val(
            imgsz=imgsz_val,
            batch=batch_val,
            workers=0,
            save_json=save_json,
            conf=conf,
            iou=iou,
            max_det=max_det,
            name=f"{run_label}_val",
            split="val",
            project="runs/detect"
        )
        write_results(run_label, val_metrics, split="val")

        # Final test set evaluation
        test_metrics = model.val(
            imgsz=imgsz_val,
            batch=batch_val,
            workers=0,
            save_json=save_json,
            conf=conf,
            iou=iou,
            max_det=max_det,
            name=f"{run_label}_test",
            split="test",
            project="runs/detect"
        )
        write_results(run_label, test_metrics, split="test")

    else:
        print("❌ Final model weight file 'last.pt' does not exist.")

# ✅ Run the training
run_yolo_training()
