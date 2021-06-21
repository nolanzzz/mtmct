

import os.path as osp
import os
import cv2
from PIL import Image
import glob
from rectpack import newPacker

import random

def draw_packed_reid_image(image_folder
                           ,reid_dataset_folder
                           ,output_path
                           ,without_distractors
                           ,take_count
                           ,pack_image_dims=(1024,512)
                           ):
    def add_border(image, border_in_px=3):

        width,height = image.size
        new_width, new_height = width + border_in_px, height + border_in_px

        new_image = Image.new('RGB', (new_width, new_height), color='white')
        center_x = int((new_width - width) / 2)
        center_y = int((new_height - height) / 2)
        new_image.paste(image, (center_x, center_y))

        return new_image

    image_names = sorted(os.listdir(image_folder))

    if without_distractors:
        distractors = getAllDistractors(reid_dataset_folder)
        image_names = set(image_names)
        image_names = list(image_names - distractors)
        image_names = sorted(image_names)



    random.seed(574)
    image_names = random.choices(image_names,k=take_count)

    images = []
    packer = newPacker(rotation=False)
    for image_id, image_name in enumerate(image_names):
        image_path = os.path.join(image_folder,image_name)
        img = Image.open(image_path)

        img = add_border(image=img,border_in_px=3)
        images.append(img)
        width, height = img.size

        packer.add_rect(width=width,height=height,rid=image_id)


    # Add the bins where the rectangles will be placed

    packer.add_bin(width=pack_image_dims[0],height=pack_image_dims[1],count=1)

    # Start packing
    packer.pack()

    bin = packer[0]
    print("bin len {}".format(len(bin)))

    img = Image.new('RGB', (pack_image_dims[0], pack_image_dims[1]), color = 'white')

    for rect in bin:
        w = rect.width
        h = rect.height
        # rect is a Rectangle object
        x = rect.x  # rectangle bottom-left x coordinate
        y = pack_image_dims[1] - rect.y - h# rectangle bottom-left y coordinate

        rid = rect.rid
        image = images[rid]
        img.paste(image, (x, y))

    img.show()
    img.save(output_path)


def getAllDistractors(reid_dataset_folder):
    def getDistractors(testOrTrainSuffix):
        distractor_imgs_path = osp.join(reid_dataset_folder, osp.join("distractors/", testOrTrainSuffix))
        if not osp.exists(distractor_imgs_path):
            return set()
        img_names = [osp.basename(img_path) for img_path in glob.glob(osp.join(distractor_imgs_path, '*.png'))]

        return set(img_names)

    train_img_names = getDistractors("train")
    test_img_names = getDistractors("test")
    query_img_names = getDistractors("query")

    return train_img_names.union(test_img_names, query_img_names)


def are_there_distractors_in_query(reid_dataset_folder):
    distractors = getAllDistractors(reid_dataset_folder=reid_dataset_folder)

    query_filenames = set(os.listdir(osp.join(reid_dataset_folder,"query/")))

    print(query_filenames.intersection(distractors))


if __name__ == "__main__":
    draw_packed_reid_image(reid_dataset_folder="/media/philipp/philippkoehl_ssd/reid_camid_GTA_22.07.2019/"
                            ,image_folder="/media/philipp/philippkoehl_ssd/reid_camid_GTA_22.07.2019/train"
                           ,output_path="/media/philipp/philippkoehl_ssd/packed_reid_image.jpg"
                           ,without_distractors=True
                           ,take_count=270)
