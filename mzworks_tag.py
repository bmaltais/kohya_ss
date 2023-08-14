import argparse
import pathlib


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dir", default="./dataset/train")
    parser.add_argument("--update", default="")
    parser.add_argument("--append", default="")
    parser.add_argument("--remove", default="")
    parser.add_argument("--dry-run", action="store_true")

    args = parser.parse_args()

    o_path = pathlib.Path(args.dir)

    dict_caption = {}
    dict_tag = {}

    for o_pathname in o_path.glob("*.caption"):
        pathname = str(o_pathname)

        with open(pathname, "r") as rf:
            if args.update != "":
                list_tag = [v.strip() for v in args.update.split(",")]
            else:
                list_tag = [v.strip() for v in rf.read().split(",")]

            dict_caption[pathname] = list_tag
            for tag in list_tag:
                try:
                    dict_tag[tag] += 1
                except KeyError:
                    dict_tag[tag] = 1

    if args.append != "":
        for pathname, list_tag in dict_caption.items():
            if args.append in list_tag:
                list_tag.remove(args.append)

            list_tag.insert(0, args.append)

    if args.remove != "":
        for pathname, list_tag in dict_caption.items():
            if args.remove in list_tag:
                list_tag.remove(args.remove)

    if args.dry_run is True:
        for pathname, list_tag in dict_caption.items():
            print("{:32s} : {:s}".format(pathname, ", ".join(list_tag)))
    else:
        for pathname, list_tag in dict_caption.items():
            with open(pathname, "w") as wf:
                wf.write(", ".join(list_tag))


if __name__ == "__main__":
    main()
