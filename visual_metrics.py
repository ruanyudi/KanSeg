import argparse
import json
import matplotlib.pyplot as plt

METRICS=['total_loss','segm/AP','bbox/AP','segm/AP50','bbox/AP50']

def get_parser():
    parser = argparse.ArgumentParser(description="maskdino demo for builtin configs")
    parser.add_argument(
        "--metrics-file",
        default="./output/metrics.json",
        metavar="FILE",
        help="path to metric file",
    )
    return parser

def show_metric():
    for METRIC in METRICS:
        metric2show = []
        for metric in AP_metrics:
            metric2show.append(metric[METRIC])
        plt.plot(metric2show,label=METRIC)
    plt.legend()
    plt.savefig('visual_metric.png',dpi=300)

if __name__ == '__main__':
    args = get_parser().parse_args()
    AP_metrics = []
    for line in open(args.metrics_file,'r'):
        metrics = json.loads(line)
        if "segm/AP" in metrics.keys():
            AP_metrics.append(metrics)
    print(AP_metrics[0].keys())
    show_metric()

    