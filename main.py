import os
import torch
import torch.nn as nn
import torchvision.models as models
from ultralytics import YOLO
from thop import profile
import copy

class CustomBackbone(nn.Module):
    """EfficientNet-B0 Backbone for YOLOv8 with Feature Map Extraction and Channel Adjustment"""
    
    def __init__(self, backbone_type='efficientnet', **kwargs):
        super().__init__()
        
        if backbone_type == 'efficientnet':
            self.backbone = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
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
        print("🔍 Extracting EfficientNet Feature Layers...", flush=True)
    
        layers = {
            'P3': self.backbone.features[:4],   # Expected output: [*, 40, 38, 38]
            'P4': self.backbone.features[4:6],    # Expected output: [*, 112, 19, 19]
            'P5': self.backbone.features[6:]      # Expected output: [*, 1280, 10, 10]
        }
    
        for name, layer in layers.items():
            print(f"🔹 {name} Layer Extracted: {layer}", flush=True)
    
        return nn.ModuleDict(layers)

    def forward(self, x):
        print("🟢 Inside CustomBackbone forward()", flush=True)
        # Compute raw feature maps sequentially:
        p3 = self.feature_layers['P3'](x)
        print(f"✅ P3 raw: {p3.shape}", flush=True)  # Expected: [batch, 40, 38, 38]
        
        p4 = self.feature_layers['P4'](p3)
        print(f"✅ P4 raw: {p4.shape}", flush=True)  # Expected: [batch, 112, 19, 19]
        
        p5 = self.feature_layers['P5'](p4)
        print(f"✅ P5 raw: {p5.shape}", flush=True)  # Expected: [batch, 1280, 10, 10]
        
        # Adapt channels:
        p3_out = self.channel_adapter[0](p3)
        print(f"🚀 P3 after adapter: {p3_out.shape}", flush=True)
        
        p4_out = self.channel_adapter[1](p4)
        print(f"🚀 P4 after adapter: {p4_out.shape}", flush=True)
        
        p5_out = self.channel_adapter[2](p5)
        print(f"🚀 P5 after adapter: {p5_out.shape}", flush=True)
        
        return [p3_out, p4_out, p5_out]

class BackboneWrapper(nn.Module):
    """
    A wrapper to use the CustomBackbone for profiling.
    Instead of returning a list of feature maps, it returns only the last one (p5_out).
    """
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
    def forward(self, x):
        outputs = self.backbone(x)
        # Return only the final feature map for GFLOPs counting
        return outputs[-1]

def modify_model_for_efficientnet(yolo_model):
    """Replace YOLO's default backbone with EfficientNet-B0"""
    custom_backbone = CustomBackbone(backbone_type='efficientnet')

    if hasattr(yolo_model.model, 'model') and len(yolo_model.model.model) > 0:
        yolo_model.model.model[0] = custom_backbone  # Replace YOLO backbone
        print("✅ EfficientNet backbone has been successfully integrated into YOLO.", flush=True)
    else:
        print("❌ Error: YOLOv8 model structure not recognized!", flush=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    yolo_model.model = yolo_model.model.to(device)
    print("🔍 YOLO Model After Backbone Replacement:\n", yolo_model.model, flush=True)
    
    return yolo_model

def run_yolo_training():
    # SETTINGS: Update paths and hyperparameters as needed
    MODEL_PATH = "yolov8s.pt"
    NAME = 'tomatOD_run_effnet'
    DATA = 'tomatOD_yolo/data.yaml'  
    EPOCHS = 5
    BATCH = 16
    IMGSZ = 300
    LR0 = 0.01
    LRF = 0.0001
    MOMENTUM = 0.9
    OPTIMIZER = 'SGD'
    SAVE_PERIOD = 1
    VAL = True

    # Load YOLO model and integrate the EfficientNet backbone
    model = YOLO(MODEL_PATH)
    model = modify_model_for_efficientnet(model)

    # Calculate total model parameters
    print("🔍 Calculating Model Parameters...", flush=True)
    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"✅ Total Model Parameters: {total_params:.2f}M", flush=True)
    
    # --- Compute GFLOPs on a separate backbone copy using a wrapper ---
    print("🔍 Calculating GFLOPs for the backbone...", flush=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # Create a separate instance of the backbone for profiling:
    backbone_for_profile = CustomBackbone(backbone_type='efficientnet').to(device)
    # Wrap it so that its forward returns only the last feature map:
    backbone_wrapper = BackboneWrapper(backbone_for_profile)
    dummy_input = torch.randn(1, 3, IMGSZ, IMGSZ).to(device)
    try:
        flops, params = profile(backbone_wrapper, inputs=(dummy_input,))
        print(f"✅ Backbone GFLOPs (using last feature map): {flops / 1e9:.2f} GFLOPs", flush=True)
    except Exception as e:
        print("❌ Error calculating GFLOPs:", e, flush=True)
    
    # --- Continue using the original model for further testing ---
    print("🔍 Running EfficientNet Backbone Test...", flush=True)
    dummy_input_test = torch.randn(1, 3, 300, 300).to(device)
    backbone = model.model.model[0]  # Use the integrated backbone from the YOLO model
    feature_maps = backbone(dummy_input_test)
    for i, fmap in enumerate(feature_maps):
        print(f"🟢 Final Feature Map {i+1} Shape: {fmap.shape}", flush=True)

    print("🔍 Running YOLO Debug Mode (1 epoch)...", flush=True)
    model.train(
        data=DATA,
        epochs=1,
        batch=1,
        imgsz=IMGSZ,
        workers=0,
        device=0,
        name="debug_effnet",
        optimizer=OPTIMIZER,
        verbose=True
    )

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

    print("🔍 Running YOLO Testing on Test Data...", flush=True)
    test_results = model.val(
        data=DATA,
        imgsz=IMGSZ,
        batch=BATCH,
        workers=0,
        device=0,
        name=f"{NAME}_test_results",
        split="test",
        project="runs/test"
    )

    test_output_dir = os.path.join('runs', 'test', f"{NAME}_test_results")
    os.makedirs(test_output_dir, exist_ok=True)
    
    with open(os.path.join(test_output_dir, 'test_metrics.txt'), 'w') as f:
        f.write(f"Test mAP50: {test_results.box.map50}\n")
        f.write(f"Test mAP: {test_results.box.map}\n")

    print(f"✅ Test Results Saved to {test_output_dir}", flush=True)

if __name__ == "__main__":
    run_yolo_training()