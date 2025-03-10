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

            # Update these based on actual EfficientNet outputs:
            # P3 raw: 40 channels, P4 raw: 112 channels, P5 raw: 1280 channels
            self.eff_out_channels = [40, 112, 1280]  
            # Desired YOLO channels for P3, P4, P5
            self.yolo_out_channels = [256, 512, 1024] 

        # Extract feature layers from EfficientNet
        self.feature_layers = self._get_feature_layers()

        # Channel adapters: Map EfficientNet outputs to YOLO expected channels.
        self.channel_adapter = nn.ModuleList([
            nn.Conv2d(40, 256, kernel_size=1),    # P3: 40 -> 256
            nn.Conv2d(112, 512, kernel_size=1),    # P4: 112 -> 512
            nn.Conv2d(1280, 1024, kernel_size=1)   # P5: 1280 -> 1024
        ])

    def _get_feature_layers(self):
        """Extracts feature layers from EfficientNet"""
        print("🔍 Extracting EfficientNet Feature Layers...")
    
        layers = {
            'P3': self.backbone.features[:4],   # Responsible for P3 (should yield [*, 40, 38, 38])
            'P4': self.backbone.features[4:6],    # Responsible for P4 (should yield [*, 112, 19, 19])
            'P5': self.backbone.features[6:]      # Responsible for P5 (should yield [*, 1280, 10, 10])
        }
    
        for name, layer in layers.items():
            print(f"🔹 {name} Layer Extracted: {layer}")
    
        return nn.ModuleDict(layers)

    def forward(self, x):
        print("🟢 Inside CustomBackbone forward()")
        # Compute raw feature maps sequentially:
        p3 = self.feature_layers['P3'](x)
        print(f"✅ P3 raw: {p3.shape}")  # Expected: [batch, 40, 38, 38]
        
        p4 = self.feature_layers['P4'](p3)
        print(f"✅ P4 raw: {p4.shape}")  # Expected: [batch, 112, 19, 19]
        
        p5 = self.feature_layers['P5'](p4)
        print(f"✅ P5 raw: {p5.shape}")  # Expected: [batch, 1280, 10, 10]
        
        # Now adapt channels:
        p3_out = self.channel_adapter[0](p3)
        print(f"🚀 P3 after adapter: {p3_out.shape}")
        
        p4_out = self.channel_adapter[1](p4)
        print(f"🚀 P4 after adapter: {p4_out.shape}")
        
        p5_out = self.channel_adapter[2](p5)
        print(f"🚀 P5 after adapter: {p5_out.shape}")
        
        return [p3_out, p4_out, p5_out]


def modify_model_for_efficientnet(yolo_model):
    """Replace YOLO's default backbone with EfficientNet-B0"""
    custom_backbone = CustomBackbone(backbone_type='efficientnet')

    # Ensure YOLO model follows expected structure
    if hasattr(yolo_model.model, 'model') and len(yolo_model.model.model) > 0:
        yolo_model.model.model[0] = custom_backbone  # Replace YOLO backbone
        print("✅ EfficientNet backbone has been successfully integrated into YOLO.")
    else:
        print("❌ Error: YOLOv8 model structure not recognized!")

    # Force backbone to device
    yolo_model.model = yolo_model.model.to('cuda' if torch.cuda.is_available() else 'cpu')
    print("🔍 YOLO Model After Backbone Replacement:\n", yolo_model.model)
    
    return yolo_model


def run_yolo_training():
    # SETTINGS: Update dataset path
    MODEL_PATH = "yolov8n.pt"
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

    # Load YOLO model
    model = YOLO(MODEL_PATH)

    # Modify YOLO model to use EfficientNet
    model = modify_model_for_efficientnet(model)

    # Debugging: Check backbone outputs with dummy input before training
    print("🔍 Running EfficientNet Backbone Test...")
    dummy_input = torch.randn(1, 3, 300, 300).to('cuda' if torch.cuda.is_available() else 'cpu')
    backbone = model.model.model[0]  # Get Custom Backbone
    feature_maps = backbone(dummy_input)
    for i, fmap in enumerate(feature_maps):
        print(f"🟢 Final Feature Map {i+1} Shape: {fmap.shape}")

    # Run YOLO training in debug mode for 1 epoch
    print("🔍 Running YOLO Debug Mode (1 epoch)...")
    model.train(
        data=DATA,
        epochs=1,  # Debug run with 1 epoch
        batch=1,   # Small batch size
        imgsz=IMGSZ,
        workers=0,
        device=0,
        name="debug_effnet",
        optimizer=OPTIMIZER,
        verbose=True
    )

    # Run full training
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

# ✅ **Testing after training (Validation on test set)**
    print("🔍 Running YOLO Testing on Test Data...")
    test_results = model.val(
        data=DATA,
        imgsz=IMGSZ,
        batch=BATCH,
        workers=0,
        device=0,
        name=f"{NAME}_test_results",
        split="test",  # ✅ Use the test split for evaluation
        project="runs/test"
    )

    # Save test results
    test_output_dir = os.path.join('runs', 'test', f"{NAME}_test_results")
    os.makedirs(test_output_dir, exist_ok=True)
    
    with open(os.path.join(test_output_dir, 'test_metrics.txt'), 'w') as f:
        f.write(f"Test mAP50: {test_results.box.map50}\n")
        f.write(f"Test mAP: {test_results.box.map}\n")

    print(f"✅ Test Results Saved to {test_output_dir}")

# Run the training
run_yolo_training()
