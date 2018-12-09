import os
import hashlib
import pickle


def main():
    root_path = "/home/saliency/waterloo/Castor-data/datasets/WikiQA/"
    train_path = "train/"
    test_path = "test/"
    dev_path = "dev/"
    file_path = [root_path + train_path, root_path + test_path, root_path + dev_path]
    toks_to_parent = dict()

    for _dir in file_path:
        os.chdir(_dir)
        f_toks = open("a.toks").readlines()
        f_parent = open("a.parents").readlines()
        for i in range(len(f_toks)):
            tok_line = f_toks[i].rstrip(".\n")
            code = hashlib.sha224(tok_line.encode()).hexdigest()
            if f_toks[i][-2] == ".":
                reduced_fparent = ' '.join(f_parent[i].strip("\n").split()[:-1])
                toks_to_parent[code] = reduced_fparent.split()
            else:
                toks_to_parent[code] = f_parent[i].strip("\n").split()

        f_toks_b = open("b.toks").readlines()
        f_parent_b = open("b.parents").readlines()
        for i in range(len(f_toks_b)):
            tok_line_b = f_toks_b[i].rstrip(".\n")
            code = hashlib.sha224(tok_line_b.encode()).hexdigest()
            try:
                if f_toks_b[i][-2] == ".":
                    reduced_fparent_b = ' '.join(f_parent_b[i].strip("\n").split()[:-1])
                    toks_to_parent[code] = reduced_fparent_b.split()
                else:
                    toks_to_parent[code] = f_parent_b[i].strip("\n").split()
            except IndexError:
                print("f_toks: ", f_toks_b[i])
                print("f_parent: ", f_parent_b[i])
                print("error!")

    with open("/home/saliency/waterloo/Castor/parent_tree_wiki.pkl", "wb") as f:
        pickle.dump(toks_to_parent, f)


if __name__ == "__main__":
    main()
