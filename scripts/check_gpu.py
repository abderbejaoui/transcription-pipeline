import sys
try:
    import torch
except Exception as e:
    print("ERROR: could not import torch:", e)
    sys.exit(2)


def main():
    print("torch.__version__:", getattr(torch, "__version__", "unknown"))
    cuda_ok = torch.cuda.is_available()
    print("torch.cuda.is_available():", cuda_ok)
    print("torch.cuda.device_count():", torch.cuda.device_count())
    if cuda_ok:
        try:
            idx = torch.cuda.current_device()
            print("torch.cuda.current_device():", idx)
            try:
                print("torch.cuda.get_device_name(0):", torch.cuda.get_device_name(0))
            except Exception as e:
                print("device name error:", e)
            # allocate a modest tensor on CUDA to force memory use
            x = torch.randn(512, 512).to('cuda')
            print("allocated tensor device:", x.device)
            print("torch.cuda.memory_allocated():", torch.cuda.memory_allocated())
            del x
        except Exception as e:
            print("CUDA operation error:", e)
    else:
        print("CUDA not available — process will run on CPU.")


if __name__ == '__main__':
    main()
