# 校园导航图片资源目录

此目录用于存放校园导航工具的地图图片和目的地实景照片。

## 目录结构

```
assets/
├── maps/          # 地图图片（标注路线）
│   ├── yuelu_academy_map.png      # 岳麓书院地图
│   ├── fulinshe_map.png           # 复临舍地图
│   ├── zonghe_building_map.png    # 综合楼地图
│   ├── library_map.png            # 图书馆地图
│   ├── canteen1_map.png           # 一食堂地图
│   ├── canteen2_map.png           # 二食堂地图
│   ├── dezhi_dorm_map.png         # 德智公寓地图
│   ├── tianma_dorm_map.png        # 天马公寓地图
│   ├── east_red_square_map.png    # 东方红广场地图
│   ├── hospital_map.png           # 校医院地图
│   ├── stadium_map.png            # 体育场地图
│   └── grad_school_map.png        # 研究生院地图
│
└── photos/        # 目的地实景照片
    ├── yuelu_academy_1.jpg
    ├── yuelu_academy_2.jpg
    └── ...
```

## 图片配置说明

在 `tools/campus_navigate.py` 中的 `LOCATIONS_DB` 字典里，每个地点可以配置以下图片相关字段：

```python
"地点名称": {
    "area": "所在区域",
    "description": "地点描述",
    "nearby": "周边设施",
    "route": "路线指引",
    "map_image": "assets/maps/xxx_map.png",  # 地图图片路径
    "photos": [                              # 实景照片列表
        "assets/photos/xxx_1.jpg",
        "assets/photos/xxx_2.jpg"
    ],
}
```

## 使用方式

1. 将地图图片放入 `maps/` 目录
2. 将实景照片放入 `photos/` 目录
3. 在 `campus_navigate.py` 中取消注释对应的图片路径配置
4. Display模块会自动读取并展示这些图片

## 图片建议规格

- **地图图片**: 建议 800x600 像素，PNG格式，标注清晰的路线
- **实景照片**: 建议 1200x800 像素，JPG格式，展示目的地外观

## 待添加素材清单

- [ ] 岳麓书院地图 + 实景照片
- [ ] 复临舍地图 + 实景照片
- [ ] 综合楼地图 + 实景照片
- [ ] 图书馆地图 + 实景照片
- [ ] 一食堂地图 + 实景照片
- [ ] 二食堂地图 + 实景照片
- [ ] 德智公寓地图 + 实景照片
- [ ] 天马公寓地图 + 实景照片
- [ ] 东方红广场地图 + 实景照片
- [ ] 校医院地图 + 实景照片
- [ ] 体育场地图 + 实景照片
- [ ] 研究生院地图 + 实景照片
