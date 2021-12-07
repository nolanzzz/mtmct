# MTMCT
This is an implementation of a Multi-Target Multi-Camera Tracking (MTMCT) solution.
Pipeline of our solution:
<img src="readme_files/pipeline_white.png" style="zoom:35%;" />

## Tracking performance
### Results and comparisons with FairMOT and WDA Tracker trained and tested on a 6x2-minute MTA dataset
<table>
    <thead>
        <tr>
            <th rowspan=2>Method</th>
            <th colspan=5>Single-Camera</th>
            <th colspan=5>Multi-Camera</th>
        </tr>
        <tr>
            <th>MOTA</th>
            <th>IDF1</th>
            <th>IDs</th>
            <th>MT</th>
            <th>ML</th>
            <th>MOTA</th>
            <th>IDF1</th>
            <th>IDs</th>
            <th>MT</th>
            <th>ML</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>WDA</td>
            <td>58.2</td>
            <td>37.3</td>
            <td>534.2</td>
            <td>16.8%</td>
            <td>17.2</td>
            <td>46.6</td>
            <td>19.8</td>
            <td>563.8</td>
            <td>6.5%</td>
            <td>7.0%</td>
        </tr>
        <tr>
            <td>FairMOT</td>
            <td>64.1</td>
            <td><strong>48.0</strong></td>
            <td>588.2</td>
            <td>34.7%</td>
            <td>7.8%</td>
            <td>N/A</td>
            <td>N/A</td>
            <td>N/A</td>
            <td>N/A</td>
            <td>N/A</td>
        </tr>
        <tr>
            <td>Ours</td>
            <td><strong>70.8</strong></td>
            <td>47.8</td>
            <td><strong>470.2</strong></td>
            <td><strong>40.5%</strong></td>
            <td><strong>5.6%</strong></td>
            <td><strong>65.6</strong></td>
            <td><strong>31.5</strong></td>
            <td><strong>494.5</strong></td>
            <td><strong>31.2%</strong></td>
            <td><strong>1.1%</strong></td>
        </tr>
    </tbody>
</table>

### Video demos on Multi Camera Track Auto (MTA) dataset
<img src="readme_files/cam_0.gif" width="400"/> <img src="readme_files/cam_1.gif" width="400"/> <img src="readme_files/cam_2.gif" width="400"/> <img src="readme_files/cam_3.gif" width="400"/> <img src="readme_files/cam_4.gif" width="400"/> <img src="readme_files/cam_5.gif" width="400"/>

## Installation
```shell
conda create -n mtmct python=3.7.7 -y
conda activate mtmct
pip install -r requirements.txt
```
Install dependencies for FairMOT:
```shell
cd trackers/fair
conda install pytorch==1.7.0 torchvision==0.8.0 cudatoolkit=10.2 -c pytorch
pip install cython
pip install -r requirements.txt
cd DCNv2
./make.sh
conda install -c conda-forge ffmpeg
```

## Download data
Go to [https://github.com/schuar-iosb/mta-dataset](https://github.com/schuar-iosb/mta-dataset) to download the MTA data. Or use other datasets that match the same format.

## Configurations
Modify config files under `tracker_configs` and `clustering_configs` for customization. More instructions can be found at [ifzhang/FairMOT](https://github.com/ifzhang/FairMOT) and [koehlp/wda_tracker](https://github.com/koehlp/wda_tracker).
E.g. in `configs/tracker_configs/fair_high_20e` set the data -> source -> base_folder to your dataset location.

## Tracking
Run single and the multi-camera tracking with one script:
```shell
sh start.sh fair_high_20e
```

Modify config files under `tracker_configs` and `clustering_configs` for customization. More instructions can be found at [ifzhang/FairMOT](https://github.com/ifzhang/FairMOT) and [koehlp/wda_tracker](https://github.com/koehlp/wda_tracker).

## Acknowledgement
A large part of the code is borrowed from [ifzhang/FairMOT](https://github.com/ifzhang/FairMOT) and [koehlp/wda_tracker](https://github.com/koehlp/wda_tracker). The dataset used is [schuar-iosb/mta-dataset](https://github.com/schuar-iosb/mta-dataset)