import ee

ee.Initialize(project='coloque-aqui-o-nome-do-teu-projeto-cloud')

roi = ee.Geometry.Point([-54.93, -2.64]).buffer(50000)

def mask_s2_clouds(image):
    qa = image.select('QA60')
    cloud_bit_mask = 1 << 10
    cirrus_bit_mask = 1 << 11
    mask = qa.bitwiseAnd(cloud_bit_mask).eq(0).And(qa.bitwiseAnd(cirrus_bit_mask).eq(0))
    return image.updateMask(mask).divide(10000)

def add_water_mask(image):
    mndwi = image.normalizedDifference(['B3', 'B11']).rename('MNDWI')
    water_mask = mndwi.gt(0).rename('water_mask')
    return image.addBands(water_mask)

dataset_with_masks = (
    ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
    .filterBounds(roi)
    .filterDate('2023-01-01', '2023-12-31') # Coloquei 1 ano inteiro para pegar uma boa mediana
    .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))
    .map(mask_s2_clouds)
    .map(add_water_mask)
)

imagem_final = dataset_with_masks.median().clip(roi)

imagem_exportar = imagem_final.select(['B2', 'B3', 'B4', 'B8', 'water_mask'])

tarefa = ee.batch.Export.image.toDrive(
    image=imagem_exportar,
    description='Dataset_Sentinel2_Agua_Belterra',
    folder='Imagens_UNet',                
    fileNamePrefix='recorte_regiao_01',   
    scale=10,                             
    region=roi.getInfo()['coordinates'],  
    fileFormat='GeoTIFF',                 
    maxPixels=1e13                        
)

tarefa.start()