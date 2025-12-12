import ee

def ETp_list(season_dates, aoi):
    ETp_list = []
    for i in range(len(season_dates)-1):
        ETp_day = ETp(season_dates[i], season_dates[i+1], aoi)
        ETp_list.append(ETp_day)

    return ETp_list


def ETp(date1, date2, aoi):
    image = ee.ImageCollection('ECMWF/ERA5_LAND/DAILY_AGGR')\
        .filter(ee.Filter.date(date1, date2)).first()

    Pi = ee.Number(3.14)

    # AIR TEMPERATURE [K]
    tair_c = image.select('temperature_2m')\
        .rename('AirT_G')

    # WIND SPEED [M S-1]
    wind_u = image.select('u_component_of_wind_10m')

    wind_v = image.select('v_component_of_wind_10m')

    # TODO: CGM check if the select calls are needed
    wind_med = wind_u.expression(
            'sqrt(ux_u ** 2 + ux_v ** 2)', {'ux_u': wind_u, 'ux_v': wind_v},
        ).rename('ux_G')

    wind_med = wind_med.expression(
        'ux * (4.87) / log(67.8 * z - 5.42)', {'ux': wind_med, 'z': 10.0}).rename('ux_G')
    # PRESSURE [PA] CONVERTED TO KPA
    tdp = image.select('dewpoint_temperature_2m')\
        .rename('tdp')

    # ACTUAL VAPOR PRESSURE [KPA]
    ea = tdp.expression(
        '0.6108 * (exp((17.27 * T_air) / (T_air + 237.3)))',{
        'T_air': tdp.subtract(273.15)}).rename('ea')

    # SATURATED VAPOR PRESSURE [KPA]
    esat = tair_c.expression(
        '0.6108 * (exp((17.27 * T_air) / (T_air + 237.3)))', {'T_air': tair_c.subtract(273.15)}).rename('esat')

    # RELATIVE HUMIDITY (%)
    rh = ea.divide(esat).multiply(100).rename('RH_G')

    # Resample
    temp = tair_c.subtract(273.15).resample('bilinear')
    wind_speed = wind_med.resample('bilinear')
    rh = rh.resample('bilinear')
    solar_radiation = image.select('surface_solar_radiation_downwards_sum').divide(1000000).rename('solar_radiation')

    # Constants
    psychrometric_constant = 0.067  # Approximate value of psychrometric constant in kPa/°C

    # Calculating slope of the vapor pressure curve (Delta)
    delta = image.expression(
    '4098 * saturation_vapor_pressure / (temperature + 237.3)**2',{
        'saturation_vapor_pressure' : esat,
        'temperature': temp}).rename('Delta')


    # # Calculating ETp using Penman-Monteith equation
    # numerator = 0.408 * delta * (net_radiation) + psychrometric_constant * (900 / (temperature + 273)) * wind_speed * (saturation_vapor_pressure - actual_vapor_pressure)
    ETp = image.expression(
    '(0.408 * delta * net_radiation + psychrometric_constant * (900 / (temperature + 273)) * wind_speed * (saturation_vapor_pressure - actual_vapor_pressure)) / (delta + psychrometric_constant * (1 + 0.34 * wind_speed))',{
        'delta' : delta,
        'net_radiation' : solar_radiation,
        'psychrometric_constant' : psychrometric_constant,
        'saturation_vapor_pressure' : esat,
        'actual_vapor_pressure' : ea,
        'temperature' : temp,
        'wind_speed': wind_speed}).rename('ETp')

    #ADD BANDS
    ETp_mean = ETp.reduceRegion(
                    reducer = ee.Reducer.mean(),
                    geometry = aoi,
                    scale = 30,
                    maxPixels = 9e14
                    ).getInfo()

    return ETp_mean['ETp']
