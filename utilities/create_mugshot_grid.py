

import numpy as np
import cv2
import os

def create_mugshot_grid(mugshot_folder, shape,remove_indices, output_path):

    mugshot_list = os.listdir(mugshot_folder)
    mugshot_list.sort()
    mugshot_list = [ filename for i,filename in enumerate(mugshot_list) if i not in remove_indices ]




    rows = []
    row = []
    image_no = 0
    for row_no in range(shape[0]):
        for column_no in range(shape[1]):
            image_name = mugshot_list[image_no]
            image_no += 1

            image_path = os.path.join(mugshot_folder,image_name)

            img = cv2.imread(image_path)
            x = 670
            y = 317
            w = 285
            h = 725
            crop_img = img[y:y + h, x:x + w]

            row.append(crop_img)

        row_concatenated = np.concatenate(row, axis=1)
        row = []
        rows.append(row_concatenated)


    vis = np.concatenate(rows, axis=0)
    resized_image = cv2.resize(vis, (0,0), fx=0.7, fy=0.7)

    os.makedirs(os.path.split(output_path)[0],exist_ok=True)
    cv2.imwrite(filename=output_path
                ,img=resized_image
                ,params=[int(cv2.IMWRITE_JPEG_QUALITY), 90])

    cv2.imshow("imshow window", resized_image)
    cv2.waitKey(0)



if __name__ == "__main__":

    create_mugshot_grid(mugshot_folder="/media/philipp/philippkoehl_ssd/GTA_mugshots/front_"
                        ,shape=(3,10)
                        ,remove_indices=[3,16,20]
                        ,output_path="/media/philipp/philippkoehl_ssd/work_dirs/visualizations/mugshot_grid.jpg")