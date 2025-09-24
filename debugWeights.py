import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from collections import defaultdict
import V14_varLen_simpleRepeat


def summarize_weights_by_type(model):
    # Group parameters by layer type
    # layer_groups = defaultdict(list)

    for module in model.children():
        if isinstance(module,V14_varLen_simpleRepeat.Multiply)==True:
            continue
        # w = module.weight.data
        if isinstance(module,V14_varLen_simpleRepeat.ConvLayers)==True:
            for submodule in module.children():
                if hasattr(submodule, "weight") and submodule.weight is not None:
                    w = submodule.weight.data
                    print(f"{type(submodule).__name__}.weight shape={w.shape}, "
                          f"mean={w.mean().item():.4f}, std={w.std().item():.4f}")

                    # histogram
                    plt.figure(figsize=(6, 3))
                    plt.hist(w.detach().cpu().numpy().flatten(), bins=50)
                    plt.title(f"{type(submodule).__name__}.weight")
                    plt.show()

                if hasattr(submodule, "bias") and submodule.bias is not None:
                    b = submodule.bias.data
                    print(f"{type(submodule).__name__}.bias shape={b.shape}, "
                          f"mean={b.mean().item():.4f}, std={b.std().item():.4f}")

                    plt.figure(figsize=(6, 3))
                    plt.hist(b.detach().cpu().numpy().flatten(), bins=50)
                    plt.title(f"{type(submodule).__name__}.bias")
                    plt.show()

        if hasattr(module, "weight") and module.weight is not None:
            w = module.weight.data
            print(f"{type(module).__name__}.weight shape={w.shape}, "
                  f"mean={w.mean().item():.4f}, std={w.std().item():.4f}")

            # histogram
            plt.figure(figsize=(6, 3))
            plt.hist(w.detach().cpu().numpy().flatten(), bins=50)
            plt.title(f"{type(module).__name__}.weight")
            plt.show()

        if hasattr(module, "bias") and module.bias is not None:
            b = module.bias.data
            print(f"{type(module).__name__}.bias shape={b.shape}, "
                  f"mean={b.mean().item():.4f}, std={b.std().item():.4f}")

            plt.figure(figsize=(6, 3))
            plt.hist(b.detach().cpu().numpy().flatten(), bins=50)
            plt.title(f"{type(module).__name__}.bias")
            plt.show()
        print("!!!")

    # for name, module in model.named_modules():
    #     if len(list(module.children())) == 0:  # leaf modules only
    #         layer_groups[type(module).name].append((name, module))
    #
    # for layer_type, modules in layer_groups.items():
    #     print(f"\n=== {layer_type} ===")
    #     for name, module in modules:
    #         if hasattr(module, "weight") and module.weight is not None:
    #             w = module.weight.data
    #             print(f"{name}.weight shape={w.shape}, "
    #                   f"mean={w.mean().item():.4f}, std={w.std().item():.4f}")
    #
    #             # histogram
    #             plt.figure(figsize=(6, 3))
    #             plt.hist(w.detach().cpu().numpy().flatten(), bins=50)
    #             plt.title(f"{name}.weight ({layer_type})")
    #             plt.show()
    #
    #         if hasattr(module, "bias") and module.bias is not None:
    #             b = module.bias.data
    #             print(f"{name}.bias shape={b.shape}, "
    #                   f"mean={b.mean().item():.4f}, std={b.std().item():.4f}")
    #
    #             plt.figure(figsize=(6, 3))
    #             plt.hist(b.detach().cpu().numpy().flatten(), bins=50)
    #             plt.title(f"{name}.bias ({layer_type})")
    #             plt.show()