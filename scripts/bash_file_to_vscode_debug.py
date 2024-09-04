"""
"""

from argparse import ArgumentParser
from pathlib import Path


def main(file: Path) -> str:
    args = []
    add = False
    with open(file, "r") as fp:
        for line in fp:
            if add:
                args.append(line)
            if line.startswith("python"):
                add = True

    args = [a for a in args if not a.startswith("#")]
    args = [a.replace('"', "").replace("'", "").replace("\\", "").rstrip() for a in args]
    for i in range(len(args)):
        parts = args[i].split(" ")
        if len(parts) < 2:
            continue
        args[i] = parts[0] + "=" + " ".join(parts[1:])
    args = [f'"{a}"' for a in args]

    return ", ".join(args)


if __name__ == "__main__":

	parser = ArgumentParser()
	parser.add_argument("file", type=Path)
	args = parser.parse_args()

	print(main(args.file))
