import os
import shutil

class DataList:
    def __init__(self, img_path, check_path=None):
        self.img_path = img_path
        self.check_path = check_path
        self.data_list = []
        self.data_dict = dict()

    def readData(self):
        for root, dirs, files in os.walk(self.img_path):
            if 'train_data' in dirs:
                self.data_list.append(root)
        for name in self.data_list:
            for root, dirs, files in os.walk(os.path.join(name, "train_data")):
                self.data_dict[root] = files
        # for i in os.listdir(self.img_path):
        #     self.data_list.append(i)

    def writeTxt(self):
        for root, imgs in self.data_dict.items():
            txt_name = "{}.txt".format(os.path.basename(root))
            print(root.replace(root.split('/')[-1], ""))
            with open(os.path.join(root.replace(root.split('/')[-1], ""), txt_name), 'w', encoding='UTF-8') as f:
                for img in imgs:
                    f.write(img + '\n')
            f.close()
            print(root)

    def data_check(self):
        for i in self.data_list:
            if i in os.listdir(self.check_path):
                print(i)
                os.remove(os.path.join(self.img_path, i))

    def main(self):
        self.readData()
        self.writeTxt()
        # self.data_check()


d = DataList(img_path='/data/DATA/Mower_labeled_data/Occlusion_data/occulation_datasets/wpy/occulation_data/train_data/val/no_occu', check_path='/data/DATA/Mower_labeled_data/Occlusion_data/occulation_datasets/wpy/occulation_data/transfer_occu_out/val/transfer_occu')
d.main()
