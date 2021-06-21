

from utilities.pandas_loader import load_csv
import os
from collections import Counter

import pandas as pd

ped_type_id_to_description = {

    0 : "Player character Michael" ,
    1 : "Player character Franklin",
    2 : "Player character Trevor" ,
    29 : "Army" ,
    28 : "Animal" ,
    27 : "SWAT" ,
    21 : "Los Santos Fire Department" ,
    20 : "Paramedic" ,
    6 : "Cop" ,
    4 : "Male" ,
    5 : "Female" ,
    26 : "Human"

}


def get_ped_types(dataset_folder,cam_ids,working_dir,output_path):

    ped_type_frequency = pd.DataFrame()

    def group_and_drop_unnecessary(cam_dataframe):
        cam_dataframe = cam_dataframe.groupby(by=["person_id"],as_index=False).mean()
        cam_dataframe = cam_dataframe[["ped_type","person_id"]]
        cam_dataframe = cam_dataframe.astype(int)
        return cam_dataframe

    ped_types_all_cams = []
    for cam_id in cam_ids:
        cam_dataframe_path = os.path.join(dataset_folder,"cam_{}".format(cam_id),"coords_cam_{}.csv".format(cam_id))
        cam_dataframe = load_csv(working_dir=working_dir,csv_path=cam_dataframe_path)

        cam_dataframe = group_and_drop_unnecessary(cam_dataframe)
        ped_type_and_person_id = list(zip(cam_dataframe["ped_type"],cam_dataframe["person_id"]))
        ped_types_all_cams.extend(ped_type_and_person_id)

    ped_types_all_cams = set(ped_types_all_cams)

    ped_types_all_cams = list(map(lambda x: x[0],ped_types_all_cams))
    ped_type_counter = Counter(ped_types_all_cams)

    for key, value in ped_type_counter.items():
        desc = ped_type_id_to_description[key]
        ped_type_frequency = ped_type_frequency.append({"Ped Type Description": desc, "Frequency": value}
                                                       ,ignore_index=True)


    ped_type_frequency = ped_type_frequency[["Ped Type Description", "Frequency"]]
    ped_type_frequency = ped_type_frequency.astype({ "Frequency" : int })
    print(ped_type_frequency.to_latex(index=False))


    os.makedirs(os.path.split(output_path)[0],exist_ok=True)
    ped_type_frequency.to_csv(output_path,index=False)
    print("Saved to path: {}".format(output_path))

if __name__ == "__main__":
    get_ped_types(dataset_folder="/net/merkur/storage/deeplearning/users/koehl/gta/GTA_Dataset_22.07.2019/test"
                  ,cam_ids=list(range(6))
                  ,working_dir="/home/koehlp/Downloads/work_dirs"
                  ,output_path="/home/koehlp/Downloads/work_dirs/statistics/ped_type_frequency_gta2207_test.csv")