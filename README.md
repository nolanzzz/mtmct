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
            <td>48.0</td>
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
            <td>WDA</td>
            <td>70.8</td>
            <td>47.8</td>
            <td>470.2</td>
            <td>40.5%</td>
            <td>5.6%</td>
            <td>65.6</td>
            <td>31.5</td>
            <td>494.5</td>
            <td>31.2%</td>
            <td>1.1%</td>
        </tr>
    </tbody>
</table>


### Video demos on Multi Camera Track Auto (MTA) dataset
<img src="readme_files/cam_0.gif" width="400"/> <img src="readme_files/cam_1.gif" width="400"/> <img src="readme_files/cam_2.gif" width="400"/> <img src="readme_files/cam_3.gif" width="400"/> <img src="readme_files/cam_4.gif" width="400"/> <img src="readme_files/cam_5.gif" width="400"/>
