def entryToItem(entry: str):
    try:
        key, *value = entry.split()
        value = " ".join(value)
        return key.lower(), float(value)
    except:
        return None

def logToArr(file: str, mainKey: str = "epoch", sep: str = "|"):
    with open(file, "r") as f:
        lines = f.readlines()
    records = {}
    for line in lines:
        items = map(entryToItem, line.strip().split(sep))
        entries = dict([_ for _ in items if _ is not None])
        if not mainKey in entries: continue
        else:
            idx = entries[mainKey]
            del entries[mainKey]
            for key, value in entries.items():
                if not key in records: records[key] = []
                records[key].append((idx, value))
    return records

def save_train_log(runID: str):
    path = RUN_PATH / runID / "train" / "log.txt"
    if not isfile(path):
        print(f"Run {runID} does not have train log.")
        return
    records = logToArr(path)
    fig = plt.figure(dpi=500, figsize=(10, 5))
    ax = fig.gca()
    for key, value in records.items():
        ax.plot(*zip(*value), label=key)
    fig.legend()
    fig.savefig(RUN_PATH / runID / "train.png", transparent=True)
    print(f"Run {runID}: train log saved.")

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from argparse import ArgumentParser
    from os.path import isfile
    from lib.Context import getRunList
    from util.env import RUN_PATH
    parser = ArgumentParser()
    parser.add_argument("runID", type=str, nargs="*", help="Run ID")
    args = parser.parse_args()
    idList = args.runID if len(args.runID) else getRunList()
    for runID in idList:
        save_train_log(runID)
