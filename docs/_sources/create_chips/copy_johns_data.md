1. Filter tiles. These are covering GB all the way to Dundee.

```
['29UQR', '30UUA', '29UQS', '30UWA', '30UVA', '30UXB', '30UWB',
'30UVB', '30UXC', '30UWC', '30UVC', '30UXD', '30UWD', '30UUB',
'29UQT', '30UUC', '29UQU', '30UUD', '30UVD', '30UXE', '30UWE',
'30UVE', '30UXF', '30UWF', '30UVF', '30UXG', '30UWG', '30UVG',
'29UQV', '30UUE', '30UUF', '30UUG', '30VWH', '30VVH', '29VPC',
'30VUH', '30UYB', '31UCS', '30UYC', '31UCT', '31UDT', '30UYD',
'31UCU', '31UDU', '30UYE', '31UCV']
```

2. Filter dates
    - We need data that include cloud probability, i.e. everything until 2021/04 can be ignored (for now).
    - 20210419 is the limit
3. Filter data
    - We need:
        - `S2A_*.SAFE/GRANULE/L2A_*/IMG_DATA/R10M/*_TCI_10m.jp2` (True Color Image (RGB) 0-255)
        - `S2A_*.SAFE/GRANULE/L2A_*/IMG_DATA/R10M/*_B8.jp2` (near-infra)
        - `S2A_*.SAFE/GRANULE/L2A_*/QI_DATA/MSK_CLDPRB_20m.jp2` (cloud probability)
        - `S2A_*.SAFE/GRANULE/L2A_*/QI_DATA/MSK_CLOUDS_B00.gml` (vector cloud mask)
4. Copy data
```
Sentinel2
    ZONE
     DATE
        *_TCI_10m.jp2
        *_B8.jp2
        MSK_CLDPRB_20m.jp2
        MSK_CLOUDS_B00.jp2
```