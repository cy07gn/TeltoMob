# Dataset Providing

## Raw Dataset

The raw data, including GCT flows and Mobility flows, is now provided at : 
```
./Raw
```

- The original CSV file for GCT flow is available at: [merged_GCT.csv](Raw/merged_GCT.csv)

Here is an example:

|        Date         | Road Segment 1 | ...  | Road Segment 34 | 
|:-------------------:|:--------------:|:--------------:|:--------------|
|         ...         |    ...         |    ...         |   
| 8/29 18:30 |  449        |  ...        |   244        |   
| 8/29 18:45 |  368        |  ...        |   225        |    
| 8/29 19:00 |  344        |  ...         |   247        |  
|         ...         |    ...         |    ...         |   

- The original CSV file for Mobility flow is available at: [merged_mobility.csv](Raw/merged_mobility.csv)


|        Date         |  1_to_2(edge 1) | ... | 34_to_32(edge 84) | 
|:-------------------:|:--------------:|:--------------:|:--------------:|
|         ...         |    ...         |    ...         |    ...         |    ...        |    ...        |    ...        |
| 8/29 18:30 |      24        |    ...        |   56        |
| 8/29 18:45 |        17         |     ...        |   51        |
| 8/29 19:00 |     38         |    ...        |   53        |
|         ...            |      ...        |   ...        |

- To generate the **train/val/test datasets** for each type of GCT flow as {train,val,test}.npz, please follow the [script](https://github.com/liyaguang/DCRNN/blob/master/scripts/generate_training_data.py),
using the CSV files provided above.

## train/test/valid dataset

The train/test/val data is now provided at:
- For GCT flow:
```
./GCT_Flows
```
- Mobility-Flow is also provided at:
```
./Mobility_Flows
```

## Graph Construction
As the implementation is based on pre-calculated distances between road sections, we provided the CSV file with road section distances and IDs at: 
- GCT Flow: [Distance between connected road segments (in meters)](GCT_Flows/nodes_distance.txt). 
- The distance between Mobility Flows is corresponding to the distance of their starting nodes, 
  based on: [GPS coordinates for each road segment ID](Mobility_Flows/neighbors_manual_v7_rename.csv).

Run the [script](https://github.com/liyaguang/DCRNN/blob/master/scripts/gen_adj_mx.py) to generate the Graph Structure based on the "Road Section Distance" file provided above.

The `processed Graph Structure of Road Section Network` is available at: 
- GCT Flow: [road network structure file](GCT_Flows/adj_mat_input.pkl)
- Mobility Flow: - [road network structure file](Mobility_Flows/adj_mat_input.pkl)
