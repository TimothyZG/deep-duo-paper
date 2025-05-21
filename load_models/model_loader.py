from torchvision import models as tv_models
import timm
import torch.nn as nn
from timm.data.transforms_factory import create_transform
from timm.layers import ClassifierHead
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from load_models.transforms import augment_then_model_transform

    
def get_model_with_head(
    model_name: str,
    num_classes: int,
    source: str = "torchvision",  # or 'timm'
    tv_weights: str = "DEFAULT",
    freeze: bool = True,
    m_head: int = 1,
    keep_imagenet_head=False
):
    if source == "torchvision":
        model_name = model_name.lower()
        model_fn = getattr(tv_models, model_name)
        if isinstance(tv_weights, str):
            weights_enum = tv_models.get_model_weights(model_fn)
            if tv_weights.upper() == "DEFAULT":
                weights = weights_enum.DEFAULT
            else:
                try:
                    weights = getattr(weights_enum, tv_weights)
                except AttributeError:
                    raise ValueError(f"Invalid weight name '{tv_weights}' for model '{model_name}'. "
                                     f"Available: {[w.name for w in weights_enum]}")
        else:
            weights = None
        model = model_fn(weights=weights)
        model_transform = weights.transforms() if weights is not None else None
    elif source == "timm":
        model = timm.create_model(model_name, pretrained=True)
        model_transform = create_transform(**timm.data.resolve_data_config({}, model=model))
    else:
        raise ValueError(f"Currently only support source = 'torchvision' or 'timm', received invalid source {source}")

    if freeze:
        for param in model.parameters():
            param.requires_grad = False

    def build_new_head(in_features, num_classes, m_head):
        def enable_grad(m):
            for p in m.parameters():
                p.requires_grad = True
            return m
        if m_head > 1:
            return enable_grad(nn.Sequential(
                nn.Linear(in_features, num_classes * m_head),
                nn.Unflatten(1, (m_head, num_classes)),
            ))
        else:
            return enable_grad(nn.Linear(in_features, num_classes))


    def replace_fc(m, attr_name):
        layer = getattr(m, attr_name)
        def inject_head(in_features):
            return build_new_head(in_features, num_classes, m_head)
        if isinstance(layer, nn.Sequential):
            for i in reversed(range(len(layer))):
                if isinstance(layer[i], nn.Linear):
                    in_features = layer[i].in_features
                    layer[i] = build_new_head(in_features, num_classes, m_head)
                    # print(f"Replaced Sequential {attr_name}[{i}] with new head:\n{layer[i]}")
                    return
            print(f"Warning: No nn.Linear found in Sequential '{attr_name}'. Head not replaced.")

        elif isinstance(layer, nn.Linear):
            in_features = layer.in_features
            new_head = build_new_head(in_features, num_classes, m_head)
            setattr(m, attr_name, new_head)
            # print(f"Replaced {attr_name} with new head:\n{new_head}")
            
        elif isinstance(layer, ClassifierHead):
            if isinstance(layer.fc, nn.Linear):
                in_features = layer.fc.in_features
                layer.fc = inject_head(in_features)
                # print(f"Replaced ClassifierHead.fc in '{attr_name}' with new head:\n{layer.fc}")
            else:
                print(f"Warning: ClassifierHead.fc is not a Linear layer")

        else:
            print(f"Warning: Attribute '{attr_name}' is not a Linear or Sequential layer. Got {type(layer)}")

    if keep_imagenet_head:
        pass # Keep Original Head
    # Handle naming conventions
    elif hasattr(model, "classifier"):
        replace_fc(model, "classifier")
    elif hasattr(model, "fc"):
        replace_fc(model, "fc")
    elif hasattr(model, "head"):
        replace_fc(model, "head")
    elif hasattr(model, "heads") and hasattr(model.heads, "head"):
        replace_fc(model.heads, "head")

    transform=augment_then_model_transform(model_transform)
    return model, transform
