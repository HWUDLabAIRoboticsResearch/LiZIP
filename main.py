import argparse
import os
import platform
import subprocess
import sys

from colorama import Fore, Back, Style, init


def is_jetson():
    try:
        with open('/proc/device-tree/model', 'r') as f:
            return 'jetson' in f.read().lower()
    except OSError:
        return False


def get_version_line():
    argv = sys.argv[1:]
    mode = 'python'
    model_override = None
    for i, arg in enumerate(argv):
        if arg == '--mode' and i + 1 < len(argv):
            mode = argv[i + 1]
        if arg == '--model' and i + 1 < len(argv):
            model_override = argv[i + 1]

    if is_jetson():
        try:
            with open('/proc/device-tree/model', 'r') as f:
                device = f.read().strip().rstrip('\x00')
        except OSError:
            device = "NVIDIA Jetson"
        model = os.path.relpath(model_override or DEFAULT_MODEL_ENGINE, PROJECT_ROOT)
        return Fore.GREEN + Style.BRIGHT + f"Version: {device} (ARM64, CUDA + TensorRT)\nModel:   {model}" + Style.RESET_ALL

    sys_name = platform.system()
    arch = platform.machine()
    if model_override:
        model = model_override
    elif mode == 'cpp':
        model = os.path.relpath(DEFAULT_MODEL_BIN, PROJECT_ROOT)
    else:
        model = os.path.relpath(DEFAULT_MODEL_PTH, PROJECT_ROOT)
    return Fore.GREEN + Style.BRIGHT + f"Version: {sys_name} ({arch}, CPU)\nModel:   {model}" + Style.RESET_ALL

init()


class ColoredHelpFormatter(argparse.HelpFormatter):
    def start_section(self, heading):
        super().start_section(Fore.RED + Style.BRIGHT + heading + Style.RESET_ALL)

    def _format_action(self, action):
        text = super()._format_action(action)
        if action.option_strings:
            for opt in action.option_strings:
                text = text.replace(opt, Fore.YELLOW + opt + Style.RESET_ALL, 1)
        elif action.dest != argparse.SUPPRESS:
            text = text.replace(action.dest, Fore.CYAN + Style.BRIGHT + action.dest + Style.RESET_ALL, 1)
        return text

    def _format_usage(self, usage, actions, groups, prefix):
        text = super()._format_usage(usage, actions, groups, prefix)
        return Fore.RED + text + Style.RESET_ALL

BANNER = Fore.RED + r"""
 _      _   ____ ___ ____
| |    (_) |_  /|_ _|  _ \
| |    | |  / /  | || |_) |
| |___ | | / /_  | ||  __/
|_____||_||____||___|_|

LiZIP - Neural LiDAR Point Cloud Compression
""" + Style.RESET_ALL

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL_PTH = os.path.join(PROJECT_ROOT, "models", "grid_search", "mlp_c3_h256.pth")
DEFAULT_MODEL_BIN = os.path.join(PROJECT_ROOT, "models", "grid_search", "mlp_c3_h256.bin")
DEFAULT_MODEL_ENGINE = os.path.join(PROJECT_ROOT, "models", "jetson", "mlp_c3_h256.engine")

if is_jetson():
    CPP_EXE = os.path.join(PROJECT_ROOT, "src", "cpp", "jetson", "lizip")
else:
    CPP_EXE = os.path.join(PROJECT_ROOT, "src", "cpp", "lizip.exe")


def load_python_model(model_path):
    if model_path.endswith('.engine'):
        from src.python.trt_model import TRTPointPredictor
        return TRTPointPredictor(model_path)

    import torch
    from src import PointPredictorMLP

    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict   = checkpoint["model_state_dict"]
        context_size = checkpoint.get("context_size", 3)
        hidden_dim   = checkpoint.get("hidden_dim", 256)
    else:
        state_dict   = checkpoint
        context_size = 3
        hidden_dim   = 256
    model = PointPredictorMLP(context_size=context_size, hidden_dim=hidden_dim)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def _default_bin():
    return DEFAULT_MODEL_ENGINE if is_jetson() else DEFAULT_MODEL_BIN

def _default_pth():
    return DEFAULT_MODEL_ENGINE if is_jetson() else DEFAULT_MODEL_PTH


def cmd_encode(args):
    if args.mode == "cpp":
        if not os.path.isfile(CPP_EXE):
            print(f"[error] C++ executable not found: {CPP_EXE}")
            sys.exit(1)
        bin_path = args.model or _default_bin()
        cmd = [CPP_EXE, "e", args.input, args.output, bin_path, args.compression]
        result = subprocess.run(cmd, text=True)
        sys.exit(result.returncode)
    else:
        from src import encode_file_closed_loop
        model = load_python_model(args.model or _default_pth())
        encode_file_closed_loop(args.input, args.output, model, compression=args.compression)
        size = os.path.getsize(args.output)
        print(f"Encoded -> {args.output} ({size:,} bytes)")


def cmd_decode(args):
    if args.mode == "cpp":
        if not os.path.isfile(CPP_EXE):
            print(f"[error] C++ executable not found: {CPP_EXE}")
            sys.exit(1)
        bin_path = args.model or _default_bin()
        cmd = [CPP_EXE, "d", args.input, args.output, bin_path]
        result = subprocess.run(cmd, text=True)
        sys.exit(result.returncode)
    else:
        from src import decode_file
        model = load_python_model(args.model or _default_pth())
        decode_file(args.input, args.output, model)
        print(f"Decoded -> {args.output}")


def cmd_benchmark(args):
    script = os.path.join(PROJECT_ROOT, "benchmark", "pipeline.py")
    cmd = [sys.executable, script, "--dataset", args.dataset, "--frames", str(args.frames), "--mode", args.mode]
    if args.model:
        cmd += ["--bin", args.model]
    if args.random:
        cmd.append("--random")
    result = subprocess.run(cmd)
    sys.exit(result.returncode)


def build_parser():
    fmt = lambda prog: ColoredHelpFormatter(prog, max_help_position=36)
    parser = argparse.ArgumentParser(prog="lizip", formatter_class=fmt)
    sub = parser.add_subparsers(dest="command", metavar="<command>")
    sub.required = True

    # encode
    enc = sub.add_parser("encode", help="Compress a point cloud file", formatter_class=fmt)
    enc.add_argument("input", help="Input point cloud (.bin, .txt, .ply)")
    enc.add_argument("output", help="Output compressed file (.lizip)")
    enc.add_argument("--mode", choices=["python", "cpp"], default="python", help="Encoder backend (default: python)")
    enc.add_argument("--model", metavar="PATH", help="Model file (.pth for python, .bin for cpp)")
    enc.add_argument("--compression", choices=["zlib", "lzma"], default="zlib", help="Entropy codec (default: zlib)")

    # decode
    dec = sub.add_parser("decode", help="Decompress a .lizip file", formatter_class=fmt)
    dec.add_argument("input", help="Input compressed file (.lizip)")
    dec.add_argument("output", help="Output point cloud (.bin, .txt, .ply)")
    dec.add_argument("--mode", choices=["python", "cpp"], default="python", help="Decoder backend (default: python)")
    dec.add_argument("--model", metavar="PATH", help="Model file (.pth for python, .bin for cpp)")

    # benchmark
    bench = sub.add_parser("benchmark", help="Run the benchmarking pipeline", formatter_class=fmt)
    bench.add_argument("--dataset", choices=["nuscenes", "kitti", "argoverse"], default="nuscenes")
    bench.add_argument("--frames", type=int, default=100, help="Number of frames to benchmark (default: 100)")
    bench.add_argument("--mode", choices=["python", "cpp", "dual"], default="cpp", help="Benchmark mode (default: cpp)")
    bench.add_argument("--model", metavar="PATH", help="Path to .bin model file")
    bench.add_argument("--random", action="store_true", help="Random frame sample instead of first N")

    return parser


def main():
    print(BANNER)
    print(get_version_line())
    print()
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "encode":
        cmd_encode(args)
    elif args.command == "decode":
        cmd_decode(args)
    elif args.command == "benchmark":
        cmd_benchmark(args)


if __name__ == "__main__":
    main()
