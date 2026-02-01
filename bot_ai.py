#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
Lorentzian Classification Hybrid Trading Bot
================================================================================
هذا المشروع عبارة عن روبوت تداول هجين يقوم بـ:
1. استقبال إشارات من مؤشر Lorentzian Classification على TradingView
2. تنفيذ الصفقات تلقائياً على cTrader عبر Open API
3. إدارة رأس المال بشكل تراكمي مع رافعة مالية 1:10000

المؤشر الأصلي بلغة Pine Script محفوظ كما هو بدون أي تعديلات
================================================================================
"""

# ==============================================================================
# إعدادات تسجيل الدخول - املأ هذه الحقول بنفسك
# ==============================================================================

# TradingView Settings (اختياري - للحصول على البيانات)
TRADINGVIEW_SETTINGS = {
    "username": "",  # اسم المستخدم في TradingView
    "password": "",  # كلمة المرور
    "chart_url": "",  # رابط الرسم البياني مع المؤشر
}

# cTrader Open API Settings (مطلوب للتداول)
CTRADER_SETTINGS = {
    "client_id": "",           # Client ID من cTrader
    "client_secret": "",       # Client Secret من cTrader
    "access_token": "",        # Access Token
    "refresh_token": "",       # Refresh Token
    "account_id": "",          # رقم الحساب
    "environment": "demo",     # "demo" أو "live"
    "api_url": "https://demo.ctraderapi.com:5035",  # أو "https://live.ctraderapi.com:5035"
}

# إعدادات التداول
TRADING_SETTINGS = {
    "symbol": "EURUSD",        # زوج العملة
    "initial_capital": 100.0,  # رأس المال الأولي بالدولار
    "leverage": 10000,         # الرافعة المالية 1:10000
    "volume_lots": 0.01,       # حجم الصفقة باللوت
    "timeframe": "1m",         # فريم الدقيقة
    "use_compound": True,      # استخدام التراكم
}

# ==============================================================================
# المؤشر الأصلي بلغة Pine Script - محفوظ كما هو بدون تعديلات
# ==============================================================================

PINE_SCRIPT_INDICATOR = '''
// This source code is subject to the terms of the Mozilla Public License 2.0 at https://mozilla.org/MPL/2.0/
// ©jdehorty
// @version=5
indicator('Machine Learning: Lorentzian Classification', 'Lorentzian Classification', overlay=true, precision=4, max_labels_count=500) 

import jdehorty/MLExtensions/2 as ml
import jdehorty/KernelFunctions/2 as kernels

// ====================
// ==== Background ====
// ====================

// When using Machine Learning algorithms like K-Nearest Neighbors, choosing an
// appropriate distance metric is essential. Euclidean Distance is often used as
// the default distance metric, but it may not always be the best choice. This is
// because market data is often significantly impacted by proximity to significant
// world events such as FOMC Meetings and Black Swan events. These major economic
// events can contribute to a warping effect analogous a massive object's 
// gravitational warping of Space-Time. In financial markets, this warping effect 
// can be referred to as "Price-Time".

// To help to better account for this warping effect, Lorentzian Distance can be
// used as an alternative distance metric to Euclidean Distance. The geometry of
// Lorentzian Space can be difficult to visualize at first, and one of the best
// ways to intuitively understand it is through an example involving 2 feature
// dimensions (z=2). For purposes of this example, let's assume these two features
// are Relative Strength Index (RSI) and the Average Directional Index (ADX). In
// reality, the optimal number of features is in the range of 3-8, but for the sake
// of simplicity, we will use only 2 features in this example.

// Fundamental Assumptions:
// (1) We can calculate RSI and ADX for a given timeframe.
// (2) RSI and ADX are assumed to adhere to a Gaussian distribution in the range of 0 to 100.
// (3) The most recent RSI and ADX value can be considered the origin of a coordinate 
//     system with ADX on the x-axis and RSI on the y-axis.

// Distances in Euclidean Space:
// Measuring the Euclidean Distances of historical values with the most recent point
// at the origin will yield a distribution that resembles Figure 1 (below).

//                        [RSI]
//                          |                      
//                          |                   
//                          |                 
//                      ...:::....              
//                .:.:::••••••:::•::..             
//              .:•:.:•••::::••::••....::.            
//             ....:••••:••••••••::••:...:•.          
//            ...:.::::::•••:::•••:•••::.:•..          
//            ::•:.:•:•••••••:.:•::::::...:..   
//  0        :•:....:•••••::.:::•••::••:.....            
//           ::....:.:••••••••:•••::••::..:.          
//            .:...:••:::••••••••::•••....:          
//              ::....:.....:•::•••:::::..             
//                ..:..::••..::::..:•:..              
//                    .::..:::.....:                
//                          |            
//                          |                   
//                          |
//                          |
//                         _|_ 0        
//                          
//        Figure 1: Neighborhood in Euclidean Space

// Distances in Lorentzian Space:
// However, the same set of historical values measured using Lorentzian Distance will 
// yield a different distribution that resembles Figure 2 (below).

//                         [RSI] 
//  ::..                     |                    ..:::  
//   .....                   |                  ......
//    .••••::.               |               :••••••. 
//     .:•••••:.             |            .::.••••••.    
//         .::•••••::..      |       :..••••••..      
//            .:•••••••::.........::••••••:..         
//              ..::::••••.•••••••.•••••••:.            
//                ...:•••••••.•••••••••::.              
//                  .:..••.••••••.••••..                
//  |---------------.:•••••••••••••••••.---------------[ADX]          
//  0             .:•:•••.••••••.•••••••.                
//              .••••••••••••••••••••••••:.            
//            .:••••••••••::..::.::••••••••:.          
//          .::••••••::.     |       .::•••:::.       
//         .:••••••..        |          :••••••••.     
//       .:••••:...          |           ..•••••••:.   
//     ..:••::..             |              :.•••••••.   
//    .:•....                |               ...::.:••.  
//   ...:..                  |                   :...:••.     
//  :::.                     |                       ..::  
//                          _|_ 0
//
//       Figure 2: Neighborhood in Lorentzian Space 


// Observations:
// (1) In Lorentzian Space, the shortest distance between two points is not 
//     necessarily a straight line, but rather, a geodesic curve.
// (2) The warping effect of Lorentzian distance reduces the influence  
//     of outliers and noise.
// (3) Lorentzian Distance becomes increasingly different from Euclidean Distance 
//     as the number of nearest neighbors used for comparison increases.

// ======================
// ==== Custom Types ====
// ======================

// This section uses PineScript's new Type syntax to define important data structures
// used throughout the script.

type Settings
    float source
    int neighborsCount
    int maxBarsBack
    int featureCount
    int colorCompression
    bool showExits
    bool useDynamicExits

type Label
    int long
    int short
    int neutral

type FeatureArrays
    array<float> f1
    array<float> f2
    array<float> f3
    array<float> f4
    array<float> f5

type FeatureSeries
    float f1
    float f2
    float f3
    float f4
    float f5

type MLModel
    int firstBarIndex
    array<int> trainingLabels
    int loopSize
    float lastDistance
    array<float> distancesArray
    array<int> predictionsArray
    int prediction

type FilterSettings 
    bool useVolatilityFilter
    bool useRegimeFilter
    bool useAdxFilter
    float regimeThreshold
    int adxThreshold

type Filter
    bool volatility
    bool regime
    bool adx 

// ==========================
// ==== Helper Functions ====
// ==========================

series_from(feature_string, _close, _high, _low, _hlc3, f_paramA, f_paramB) =>
    switch feature_string
        "RSI" => ml.n_rsi(_close, f_paramA, f_paramB)
        "WT" => ml.n_wt(_hlc3, f_paramA, f_paramB)
        "CCI" => ml.n_cci(_close, f_paramA, f_paramB)
        "ADX" => ml.n_adx(_high, _low, _close, f_paramA)

get_lorentzian_distance(int i, int featureCount, FeatureSeries featureSeries, FeatureArrays featureArrays) =>
    switch featureCount
        5 => math.log(1+math.abs(featureSeries.f1 - array.get(featureArrays.f1, i))) + 
             math.log(1+math.abs(featureSeries.f2 - array.get(featureArrays.f2, i))) + 
             math.log(1+math.abs(featureSeries.f3 - array.get(featureArrays.f3, i))) + 
             math.log(1+math.abs(featureSeries.f4 - array.get(featureArrays.f4, i))) + 
             math.log(1+math.abs(featureSeries.f5 - array.get(featureArrays.f5, i)))
        4 => math.log(1+math.abs(featureSeries.f1 - array.get(featureArrays.f1, i))) +
             math.log(1+math.abs(featureSeries.f2 - array.get(featureArrays.f2, i))) +
             math.log(1+math.abs(featureSeries.f3 - array.get(featureArrays.f3, i))) +
             math.log(1+math.abs(featureSeries.f4 - array.get(featureArrays.f4, i)))
        3 => math.log(1+math.abs(featureSeries.f1 - array.get(featureArrays.f1, i))) +
             math.log(1+math.abs(featureSeries.f2 - array.get(featureArrays.f2, i))) +
             math.log(1+math.abs(featureSeries.f3 - array.get(featureArrays.f3, i)))
        2 => math.log(1+math.abs(featureSeries.f1 - array.get(featureArrays.f1, i))) +
             math.log(1+math.abs(featureSeries.f2 - array.get(featureArrays.f2, i)))

// ================  
// ==== Inputs ==== 
// ================ 

// Settings Object: General User-Defined Inputs
Settings settings = 
 Settings.new(
   input.source(title='Source', defval=close, group="General Settings", tooltip="Source of the input data"),
   input.int(title='Neighbors Count', defval=8, group="General Settings", minval=1, maxval=100, step=1, tooltip="Number of neighbors to consider"),
   input.int(title="Max Bars Back", defval=2000, group="General Settings"),
   input.int(title="Feature Count", defval=5, group="Feature Engineering", minval=2, maxval=5, tooltip="Number of features to use for ML predictions."),
   input.int(title="Color Compression", defval=1, group="General Settings", minval=1, maxval=10, tooltip="Compression factor for adjusting the intensity of the color scale."),
   input.bool(title="Show Default Exits", defval=false, group="General Settings", tooltip="Default exits occur exactly 4 bars after an entry signal. This corresponds to the predefined length of a trade during the model's training process.", inline="exits"),
   input.bool(title="Use Dynamic Exits", defval=false, group="General Settings", tooltip="Dynamic exits attempt to let profits ride by dynamically adjusting the exit threshold based on kernel regression logic.", inline="exits")
 )
   
// Trade Stats Settings
// Note: The trade stats section is NOT intended to be used as a replacement for proper backtesting. It is intended to be used for calibration purposes only.
showTradeStats = input.bool(true, 'Show Trade Stats', tooltip='Displays the trade stats for a given configuration. Useful for optimizing the settings in the Feature Engineering section.', group="General Settings")
useWorstCase = input.bool(false, "Use Worst Case Estimates", tooltip="Whether to use the worst case scenario for backtesting. This option can be useful for creating a conservative estimate that is based on close prices only, thus avoiding the effects of intrabar repainting. This option assumes that the user does not enter when the signal first appears and instead waits for the bar to close as confirmation. On larger timeframes, this can mean entering after a large move has already occurred. Leaving this option disabled is generally better for those that use this indicator as a source of confluence and prefer estimates that demonstrate discretionary mid-bar entries. Leaving this option enabled may be more consistent with traditional backtesting results.", group="General Settings")

// Settings object for user-defined settings
FilterSettings filterSettings =
 FilterSettings.new(
   input.bool(title="Use Volatility Filter", defval=true, tooltip="Whether to use the volatility filter.", group="Filters"),
   input.bool(title="Use Regime Filter", defval=true, group="Filters", inline="regime"),
   input.bool(title="Use ADX Filter", defval=false, group="Filters", inline="adx"),
   input.float(title="Threshold", defval=-0.1, minval=-10, maxval=10, step=0.1, tooltip="Whether to use the trend detection filter. Threshold for detecting Trending/Ranging markets.", group="Filters", inline="regime"),
   input.int(title="Threshold", defval=20, minval=0, maxval=100, step=1, tooltip="Whether to use the ADX filter. Threshold for detecting Trending/Ranging markets.", group="Filters", inline="adx")
 )

// Filter object for filtering the ML predictions
Filter filter =
 Filter.new(
   ml.filter_volatility(1, 10, filterSettings.useVolatilityFilter), 
   ml.regime_filter(ohlc4, filterSettings.regimeThreshold, filterSettings.useRegimeFilter),
   ml.filter_adx(settings.source, 14, filterSettings.adxThreshold, filterSettings.useAdxFilter)
  )

// Feature Variables: User-Defined Inputs for calculating Feature Series. 
f1_string = input.string(title="Feature 1", options=["RSI", "WT", "CCI", "ADX"], defval="RSI", inline = "01", tooltip="The first feature to use for ML predictions.", group="Feature Engineering")
f1_paramA = input.int(title="Parameter A", tooltip="The primary parameter of feature 1.", defval=14, inline = "02", group="Feature Engineering")
f1_paramB = input.int(title="Parameter B", tooltip="The secondary parameter of feature 2 (if applicable).", defval=1, inline = "02", group="Feature Engineering")
f2_string = input.string(title="Feature 2", options=["RSI", "WT", "CCI", "ADX"], defval="WT", inline = "03", tooltip="The second feature to use for ML predictions.", group="Feature Engineering")
f2_paramA = input.int(title="Parameter A", tooltip="The primary parameter of feature 2.", defval=10, inline = "04", group="Feature Engineering")
f2_paramB = input.int(title="Parameter B", tooltip="The secondary parameter of feature 2 (if applicable).", defval=11, inline = "04", group="Feature Engineering")
f3_string = input.string(title="Feature 3", options=["RSI", "WT", "CCI", "ADX"], defval="CCI", inline = "05", tooltip="The third feature to use for ML predictions.", group="Feature Engineering")
f3_paramA = input.int(title="Parameter A", tooltip="The primary parameter of feature 3.", defval=20, inline = "06", group="Feature Engineering")
f3_paramB = input.int(title="Parameter B", tooltip="The secondary parameter of feature 3 (if applicable).", defval=1, inline = "06", group="Feature Engineering")
f4_string = input.string(title="Feature 4", options=["RSI", "WT", "CCI", "ADX"], defval="ADX", inline = "07", tooltip="The fourth feature to use for ML predictions.", group="Feature Engineering")
f4_paramA = input.int(title="Parameter A", tooltip="The primary parameter of feature 4.", defval=20, inline = "08", group="Feature Engineering")
f4_paramB = input.int(title="Parameter B", tooltip="The secondary parameter of feature 4 (if applicable).", defval=2, inline = "08", group="Feature Engineering")
f5_string = input.string(title="Feature 5", options=["RSI", "WT", "CCI", "ADX"], defval="RSI", inline = "09", tooltip="The fifth feature to use for ML predictions.", group="Feature Engineering")
f5_paramA = input.int(title="Parameter A", tooltip="The primary parameter of feature 5.", defval=9, inline = "10", group="Feature Engineering")
f5_paramB = input.int(title="Parameter B", tooltip="The secondary parameter of feature 5 (if applicable).", defval=1, inline = "10", group="Feature Engineering")

// FeatureSeries Object: Calculated Feature Series based on Feature Variables
featureSeries = 
 FeatureSeries.new(
   series_from(f1_string, close, high, low, hlc3, f1_paramA, f1_paramB), // f1
   series_from(f2_string, close, high, low, hlc3, f2_paramA, f2_paramB), // f2 
   series_from(f3_string, close, high, low, hlc3, f3_paramA, f3_paramB), // f3
   series_from(f4_string, close, high, low, hlc3, f4_paramA, f4_paramB), // f4
   series_from(f5_string, close, high, low, hlc3, f5_paramA, f5_paramB)  // f5
 )

// FeatureArrays Variables: Storage of Feature Series as Feature Arrays Optimized
var f1Array = array.new_float()
var f2Array = array.new_float()
var f3Array = array.new_float()
var f4Array = array.new_float()
var f5Array = array.new_float()
array.push(f1Array, featureSeries.f1)
array.push(f2Array, featureSeries.f2)
array.push(f3Array, featureSeries.f3)
array.push(f4Array, featureSeries.f4)
array.push(f5Array, featureSeries.f5)

// FeatureArrays Object: Storage of the calculated FeatureArrays into a single object
featureArrays = 
 FeatureArrays.new(
  f1Array, // f1
  f2Array, // f2
  f3Array, // f3
  f4Array, // f4
  f5Array  // f5
 )

// Label Object: Used for classifying historical data as training data for the ML Model
Label direction = 
 Label.new(
   long=1, 
   short=-1, 
   neutral=0
  )

// Derived from General Settings
maxBarsBackIndex = last_bar_index >= settings.maxBarsBack ? last_bar_index - settings.maxBarsBack : 0

// EMA Settings 
useEmaFilter = input.bool(title="Use EMA Filter", defval=false, group="Filters", inline="ema")
emaPeriod = input.int(title="Period", defval=200, minval=1, step=1, group="Filters", inline="ema", tooltip="The period of the EMA used for the EMA Filter.")
isEmaUptrend = useEmaFilter ? close > ta.ema(close, emaPeriod) : true
isEmaDowntrend = useEmaFilter ? close < ta.ema(close, emaPeriod) : true
useSmaFilter = input.bool(title="Use SMA Filter", defval=false, group="Filters", inline="sma")
smaPeriod = input.int(title="Period", defval=200, minval=1, step=1, group="Filters", inline="sma", tooltip="The period of the SMA used for the SMA Filter.")
isSmaUptrend = useSmaFilter ? close > ta.sma(close, smaPeriod) : true
isSmaDowntrend = useSmaFilter ? close < ta.sma(close, smaPeriod) : true

// Nadaraya-Watson Kernel Regression Settings
useKernelFilter = input.bool(true, "Trade with Kernel", group="Kernel Settings", inline="kernel")
showKernelEstimate = input.bool(true, "Show Kernel Estimate", group="Kernel Settings", inline="kernel")
useKernelSmoothing = input.bool(false, "Enhance Kernel Smoothing", tooltip="Uses a crossover based mechanism to smoothen kernel color changes. This often results in less color transitions overall and may result in more ML entry signals being generated.", inline='1', group='Kernel Settings')
h = input.int(8, 'Lookback Window', minval=3, tooltip='The number of bars used for the estimation. This is a sliding value that represents the most recent historical bars. Recommended range: 3-50', group="Kernel Settings", inline="kernel")
r = input.float(8., 'Relative Weighting', step=0.25, tooltip='Relative weighting of time frames. As this value approaches zero, the longer time frames will exert more influence on the estimation. As this value approaches infinity, the behavior of the Rational Quadratic Kernel will become identical to the Gaussian kernel. Recommended range: 0.25-25', group="Kernel Settings", inline="kernel")
x = input.int(25, "Regression Level", tooltip='Bar index on which to start regression. Controls how tightly fit the kernel estimate is to the data. Smaller values are a tighter fit. Larger values are a looser fit. Recommended range: 2-25', group="Kernel Settings", inline="kernel")
lag = input.int(2, "Lag", tooltip="Lag for crossover detection. Lower values result in earlier entry signals.", minval=1, inline='1', group='Kernel Settings')

// Display Settings
showBarColors = input.bool(true, "Show Bar Colors", tooltip="Whether to show the bar colors.", group="Display Settings")
showBarPredictions = input.bool(defval = true, title = "Show Bar Prediction Values", tooltip = "Will show the ML model's evaluation of each bar as an integer.", group="Display Settings")
useAtrOffset = input.bool(defval = false, title = "Use ATR Offset", tooltip = "Will use the ATR offset instead of the bar prediction offset.", group="Display Settings")
barPredictionsOffset = input.float(0, "Bar Prediction Offset", minval=0, tooltip="The offset of the bar predictions as a percentage from the bar high or close.", group="Display Settings")

// =================================
// ==== Next Bar Classification ====
// =================================

// This model specializes specifically in predicting the direction of price action over the course of the next 4 bars. 
// To avoid complications with the ML model, this value is hardcoded to 4 bars but support for other training lengths may be added in the future.
src = settings.source
y_train_series = src[4] < src[0] ? direction.short : src[4] > src[0] ? direction.long : direction.neutral
var y_train_array = array.new_int(0)

// Variables used for ML Logic
var predictions = array.new_float(0)
var prediction = 0.
var signal = direction.neutral
var distances = array.new_float(0)

array.push(y_train_array, y_train_series)

// =========================
// ====  Core ML Logic  ====
// =========================

// Approximate Nearest Neighbors Search with Lorentzian Distance:
// A novel variation of the Nearest Neighbors (NN) search algorithm that ensures a chronologically uniform distribution of neighbors.

// In a traditional KNN-based approach, we would iterate through the entire dataset and calculate the distance between the current bar 
// and every other bar in the dataset and then sort the distances in ascending order. We would then take the first k bars and use their 
// labels to determine the label of the current bar. 

// There are several problems with this traditional KNN approach in the context of real-time calculations involving time series data:
// - It is computationally expensive to iterate through the entire dataset and calculate the distance between every historical bar and
//   the current bar.
// - Market time series data is often non-stationary, meaning that the statistical properties of the data change slightly over time.
// - It is possible that the nearest neighbors are not the most informative ones, and the KNN algorithm may return poor results if the
//   nearest neighbors are not representative of the majority of the data.

// Previously, the user @capissimo attempted to address some of these issues in several of his PineScript-based KNN implementations by:
// - Using a modified KNN algorithm based on consecutive furthest neighbors to find a set of approximate "nearest" neighbors.
// - Using a sliding window approach to only calculate the distance between the current bar and the most recent n bars in the dataset.

// Of these two approaches, the latter is inherently limited by the fact that it only considers the most recent bars in the overall dataset. 

// The former approach has more potential to leverage historical price action, but is limited by:
// - The possibility of a sudden "max" value throwing off the estimation
// - The possibility of selecting a set of approximate neighbors that are not representative of the majority of the data by oversampling 
//   values that are not chronologically distinct enough from one another
// - The possibility of selecting too many "far" neighbors, which may result in a poor estimation of price action

// To address these issues, a novel Approximate Nearest Neighbors (ANN) algorithm is used in this indicator.

// In the below ANN algorithm:
// 1. The algorithm iterates through the dataset in chronological order, using the modulo operator to only perform calculations every 4 bars.
//    This serves the dual purpose of reducing the computational overhead of the algorithm and ensuring a minimum chronological spacing 
//    between the neighbors of at least 4 bars.
// 2. A list of the k-similar neighbors is simultaneously maintained in both a predictions array and corresponding distances array.
// 3. When the size of the predictions array exceeds the desired number of nearest neighbors specified in settings.neighborsCount, 
//    the algorithm removes the first neighbor from the predictions array and the corresponding distance array.
// 4. The lastDistance variable is overriden to be a distance in the lower 25% of the array. This step helps to boost overall accuracy 
//    by ensuring subsequent newly added distance values increase at a slower rate.
// 5. Lorentzian distance is used as a distance metric in order to minimize the effect of outliers and take into account the warping of 
//    "price-time" due to proximity to significant economic events.

lastDistance = -1.0
size = math.min(settings.maxBarsBack-1, array.size(y_train_array)-1)
sizeLoop = math.min(settings.maxBarsBack-1, size)

if bar_index >= maxBarsBackIndex //{
    for i = 0 to sizeLoop //{
        d = get_lorentzian_distance(i, settings.featureCount, featureSeries, featureArrays) 
        if d >= lastDistance and i%4 //{
            lastDistance := d            
            array.push(distances, d)
            array.push(predictions, math.round(array.get(y_train_array, i)))
            if array.size(predictions) > settings.neighborsCount //{
                lastDistance := array.get(distances, math.round(settings.neighborsCount*3/4))
                array.shift(distances)
                array.shift(predictions)
            //}
        //}
    //}
    prediction := array.sum(predictions)
//}

// ============================
// ==== Prediction Filters ====
// ============================

// User Defined Filters: Used for adjusting the frequency of the ML Model's predictions
filter_all = filter.volatility and filter.regime and filter.adx

// Filtered Signal: The model's prediction of future price movement direction with user-defined filters applied
signal := prediction > 0 and filter_all ? direction.long : prediction < 0 and filter_all ? direction.short : nz(signal[1])

// Bar-Count Filters: Represents strict filters based on a pre-defined holding period of 4 bars
var int barsHeld = 0
barsHeld := ta.change(signal) ? 0 : barsHeld + 1
isHeldFourBars = barsHeld == 4
isHeldLessThanFourBars = 0 < barsHeld and barsHeld < 4

// Fractal Filters: Derived from relative appearances of signals in a given time series fractal/segment with a default length of 4 bars
isDifferentSignalType = ta.change(signal)
isEarlySignalFlip = ta.change(signal) and (ta.change(signal[1]) or ta.change(signal[2]) or ta.change(signal[3]))
isBuySignal = signal == direction.long and isEmaUptrend and isSmaUptrend
isSellSignal = signal == direction.short and isEmaDowntrend and isSmaDowntrend
isLastSignalBuy = signal[4] == direction.long and isEmaUptrend[4] and isSmaUptrend[4]
isLastSignalSell = signal[4] == direction.short and isEmaDowntrend[4] and isSmaDowntrend[4]
isNewBuySignal = isBuySignal and isDifferentSignalType
isNewSellSignal = isSellSignal and isDifferentSignalType

// Kernel Regression Filters: Filters based on Nadaraya-Watson Kernel Regression using the Rational Quadratic Kernel
// For more information on this technique refer to my other open source indicator located here: 
// https://www.tradingview.com/script/AWNvbPRM-Nadaraya-Watson-Rational-Quadratic-Kernel-Non-Repainting/
c_green = color.new(#009988, 20)
c_red = color.new(#CC3311, 20)
transparent = color.new(#000000, 100)
yhat1 = kernels.rationalQuadratic(settings.source, h, r, x)
yhat2 = kernels.gaussian(settings.source, h-lag, x)
kernelEstimate = yhat1
// Kernel Rates of Change
bool wasBearishRate = yhat1[2] > yhat1[1]
bool wasBullishRate = yhat1[2] < yhat1[1]
bool isBearishRate = yhat1[1] > yhat1
bool isBullishRate = yhat1[1] < yhat1
isBearishChange = isBearishRate and wasBullishRate
isBullishChange = isBullishRate and wasBearishRate
// Kernel Crossovers
bool isBullishCrossAlert = ta.crossover(yhat2, yhat1)
bool isBearishCrossAlert = ta.crossunder(yhat2, yhat1) 
bool isBullishSmooth = yhat2 >= yhat1
bool isBearishSmooth = yhat2 <= yhat1
// Kernel Colors
color colorByCross = isBullishSmooth ? c_green : c_red
color colorByRate = isBullishRate ? c_green : c_red
color plotColor = showKernelEstimate ? (useKernelSmoothing ? colorByCross : colorByRate) : transparent
plot(kernelEstimate, color=plotColor, linewidth=2, title="Kernel Regression Estimate")
// Alert Variables
bool alertBullish = useKernelSmoothing ? isBullishCrossAlert : isBullishChange
bool alertBearish = useKernelSmoothing ? isBearishCrossAlert : isBearishChange
// Bullish and Bearish Filters based on Kernel
isBullish = useKernelFilter ? (useKernelSmoothing ? isBullishSmooth : isBullishRate) : true
isBearish = useKernelFilter ? (useKernelSmoothing ? isBearishSmooth : isBearishRate) : true

// ===========================
// ==== Entries and Exits ====
// ===========================

// Entry Conditions: Booleans for ML Model Position Entries
startLongTrade = isNewBuySignal and isBullish and isEmaUptrend and isSmaUptrend
startShortTrade = isNewSellSignal and isBearish and isEmaDowntrend and isSmaDowntrend

// Dynamic Exit Conditions: Booleans for ML Model Position Exits based on Fractal Filters and Kernel Regression Filters
lastSignalWasBullish = ta.barssince(startLongTrade) < ta.barssince(startShortTrade)
lastSignalWasBearish = ta.barssince(startShortTrade) < ta.barssince(startLongTrade)
barsSinceRedEntry = ta.barssince(startShortTrade)
barsSinceRedExit = ta.barssince(alertBullish)
barsSinceGreenEntry = ta.barssince(startLongTrade)
barsSinceGreenExit = ta.barssince(alertBearish)
isValidShortExit = barsSinceRedExit > barsSinceRedEntry
isValidLongExit = barsSinceGreenExit > barsSinceGreenEntry
endLongTradeDynamic = (isBearishChange and isValidLongExit[1])
endShortTradeDynamic = (isBullishChange and isValidShortExit[1])

// Fixed Exit Conditions: Booleans for ML Model Position Exits based on a Bar-Count Filters
endLongTradeStrict = ((isHeldFourBars and isLastSignalBuy) or (isHeldLessThanFourBars and isNewSellSignal and isLastSignalBuy)) and startLongTrade[4]
endShortTradeStrict = ((isHeldFourBars and isLastSignalSell) or (isHeldLessThanFourBars and isNewBuySignal and isLastSignalSell)) and startShortTrade[4]
isDynamicExitValid = not useEmaFilter and not useSmaFilter and not useKernelSmoothing
endLongTrade = settings.useDynamicExits and isDynamicExitValid ? endLongTradeDynamic : endLongTradeStrict 
endShortTrade = settings.useDynamicExits and isDynamicExitValid ? endShortTradeDynamic : endShortTradeStrict

// =========================
// ==== Plotting Labels ====
// =========================

// Note: These will not repaint once the most recent bar has fully closed. By default, signals appear over the last closed bar; to override this behavior set offset=0.
plotshape(startLongTrade ? low : na, 'Buy', shape.labelup, location.belowbar, color=ml.color_green(prediction), size=size.small, offset=0)
plotshape(startShortTrade ? high : na, 'Sell', shape.labeldown, location.abovebar, ml.color_red(-prediction), size=size.small, offset=0)
plotshape(endLongTrade and settings.showExits ? high : na, 'StopBuy', shape.xcross, location.absolute, color=#3AFF17, size=size.tiny, offset=0)
plotshape(endShortTrade and settings.showExits ? low : na, 'StopSell', shape.xcross, location.absolute, color=#FD1707, size=size.tiny, offset=0)

// ================
// ==== Alerts ====
// ================ 

// Separate Alerts for Entries and Exits
alertcondition(startLongTrade, title='Open Long', message='LDC Open Long | {{ticker}}@{{close}} | ({{interval}})')
alertcondition(endLongTrade, title='Close Long', message='LDC Close Long | {{ticker}}@{{close}} | ({{interval}})')
alertcondition(startShortTrade, title='Open Short', message='LDC Open Short | {{ticker}}@{{close}} | ({{interval}})')
alertcondition(endShortTrade, title='Close Short', message='LDC Close Short | {{ticker}}@{{close}} | ({{interval}})')

// Combined Alerts for Entries and Exits
alertcondition(startShortTrade or startLongTrade, title='Open Position', message='LDC Open Position | {{ticker}}@{{close}} | ({{interval}})')
alertcondition(endShortTrade or endLongTrade, title='Close Position', message='LDC Close Position | {{ticker}}@[{{close}}] | ({{interval}})')

// Kernel Estimate Alerts
alertcondition(condition=alertBullish, title='Kernel Bullish Color Change', message='LDC Kernel Bullish | {{ticker}}@{{close}} | ({{interval}})')
alertcondition(condition=alertBearish, title='Kernel Bearish Color Change', message='LDC Kernel Bearish | {{ticker}}@{{close}} | ({{interval}})')

// =========================
// ==== Display Signals ==== 
// =========================

atrSpaced = useAtrOffset ? ta.atr(1) : na
compressionFactor = settings.neighborsCount / settings.colorCompression
c_pred = prediction > 0 ? color.from_gradient(prediction, 0, compressionFactor, #787b86, #009988) : prediction <= 0 ? color.from_gradient(prediction, -compressionFactor, 0, #CC3311, #787b86) : na
c_label = showBarPredictions ? c_pred : na
c_bars = showBarColors ? color.new(c_pred, 50) : na
x_val = bar_index
y_val = useAtrOffset ? prediction > 0 ? high + atrSpaced: low - atrSpaced : prediction > 0 ? high + hl2*barPredictionsOffset/20 : low - hl2*barPredictionsOffset/30
label.new(x_val, y_val, str.tostring(prediction), xloc.bar_index, yloc.price, color.new(color.white, 100), label.style_label_up, c_label, size.normal, text.align_left)
barcolor(showBarColors ? color.new(c_pred, 50) : na)

// ===================== 
// ==== Backtesting ====
// =====================

// The following can be used to stream signals to a backtest adapter
backTestStream = switch 
    startLongTrade => 1
    endLongTrade => 2
    startShortTrade => -1
    endShortTrade => -2
plot(backTestStream, "Backtest Stream", display=display.none)

// The following can be used to display real-time trade stats. This can be a useful mechanism for obtaining real-time feedback during Feature Engineering. This does NOT replace the need to properly backtest.
// Note: In this context, a "Stop-Loss" is defined instances where the ML Signal prematurely flips directions before an exit signal can be generated.
[totalWins, totalLosses, totalEarlySignalFlips, totalTrades, tradeStatsHeader, winLossRatio, winRate] = ml.backtest(high, low, open, startLongTrade, endLongTrade, startShortTrade, endShortTrade, isEarlySignalFlip, maxBarsBackIndex, bar_index, settings.source, useWorstCase)

init_table() =>
    c_transparent = color.new(color.black, 100)
    table.new(position.top_right, columns=2, rows=7, frame_color=color.new(color.black, 100), frame_width=1, border_width=1, border_color=c_transparent)

update_table(tbl, tradeStatsHeader, totalTrades, totalWins, totalLosses, winLossRatio, winRate, stopLosses) => 
    c_transparent = color.new(color.black, 100)
    table.cell(tbl, 0, 0, tradeStatsHeader, text_halign=text.align_center, text_color=color.gray, text_size=size.normal)
    table.cell(tbl, 0, 1, 'Winrate', text_halign=text.align_center, bgcolor=c_transparent, text_color=color.gray, text_size=size.normal)
    table.cell(tbl, 1, 1, str.tostring(totalWins / totalTrades, '#.#%'), text_halign=text.align_center, bgcolor=c_transparent, text_color=color.gray, text_size=size.normal)
    table.cell(tbl, 0, 2, 'Trades', text_halign=text.align_center, bgcolor=c_transparent, text_color=color.gray, text_size=size.normal)
    table.cell(tbl, 1, 2, str.tostring(totalTrades, '#') + ' (' + str.tostring(totalWins, '#') + '|' + str.tostring(totalLosses, '#') + ')', text_halign=text.align_center, bgcolor=c_transparent, text_color=color.gray, text_size=size.normal)
    table.cell(tbl, 0, 5, 'WL Ratio', text_halign=text.align_center, bgcolor=c_transparent, text_color=color.gray, text_size=size.normal)
    table.cell(tbl, 1, 5, str.tostring(totalWins / totalLosses, '0.00'), text_halign=text.align_center, bgcolor=c_transparent, text_color=color.gray, text_size=size.normal)
    table.cell(tbl, 0, 6, 'Early Signal Flips', text_halign=text.align_center, bgcolor=c_transparent, text_color=color.gray, text_size=size.normal)
    table.cell(tbl, 1, 6, str.tostring(totalEarlySignalFlips, '#'), text_halign=text.align_center, bgcolor=c_transparent, text_color=color.gray, text_size=size.normal)

if showTradeStats
    var tbl = ml.init_table()
    if barstate.islast
        update_table(tbl, tradeStatsHeader, totalTrades, totalWins, totalLosses, winLossRatio, winRate, totalEarlySignalFlips)
'''

# ==============================================================================
# Python Implementation - Trading System
# ==============================================================================

import asyncio
import json
import logging
import ssl
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
import websocket

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('trading_bot.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class SignalType(Enum):
    """أنواع الإشارات"""
    BUY = 1
    SELL = -1
    NEUTRAL = 0
    CLOSE_LONG = 2
    CLOSE_SHORT = -2


class PositionType(Enum):
    """أنواع المراكز"""
    LONG = "LONG"
    SHORT = "SHORT"
    NONE = "NONE"


@dataclass
class Trade:
    """معلومات الصفقة"""
    id: str
    symbol: str
    position_type: PositionType
    entry_price: float
    volume: float
    entry_time: datetime
    exit_price: Optional[float] = None
    exit_time: Optional[datetime] = None
    profit_loss: Optional[float] = None
    is_closed: bool = False


@dataclass
class AccountState:
    """حالة الحساب"""
    balance: float = 100.0
    equity: float = 100.0
    margin: float = 0.0
    free_margin: float = 100.0
    leverage: int = 10000
    open_trades: List[Trade] = field(default_factory=list)
    closed_trades: List[Trade] = field(default_factory=list)
    current_position: PositionType = PositionType.NONE
    cumulative_profit: float = 0.0


# ==============================================================================
# Lorentzian Classification Algorithm (Python Implementation)
# ==============================================================================

class LorentzianClassifier:
    """
    تنفيذ خوارزمية Lorentzian Classification للتعلم الآلي
    """
    
    def __init__(
        self,
        neighbors_count: int = 8,
        max_bars_back: int = 2000,
        feature_count: int = 5,
        use_volatility_filter: bool = True,
        use_regime_filter: bool = True,
        use_adx_filter: bool = False,
        regime_threshold: float = -0.1,
        adx_threshold: int = 20
    ):
        self.neighbors_count = neighbors_count
        self.max_bars_back = max_bars_back
        self.feature_count = feature_count
        self.use_volatility_filter = use_volatility_filter
        self.use_regime_filter = use_regime_filter
        self.use_adx_filter = use_adx_filter
        self.regime_threshold = regime_threshold
        self.adx_threshold = adx_threshold
        
        # Feature parameters (matching Pine Script defaults)
        self.feature_params = {
            'f1': {'type': 'RSI', 'paramA': 14, 'paramB': 1},
            'f2': {'type': 'WT', 'paramA': 10, 'paramB': 11},
            'f3': {'type': 'CCI', 'paramA': 20, 'paramB': 1},
            'f4': {'type': 'ADX', 'paramA': 20, 'paramB': 2},
            'f5': {'type': 'RSI', 'paramA': 9, 'paramB': 1},
        }
        
        # Data storage
        self.price_data: List[Dict[str, float]] = []
        self.feature_arrays: Dict[str, List[float]] = {
            'f1': [], 'f2': [], 'f3': [], 'f4': [], 'f5': []
        }
        self.y_train: List[int] = []
        
    def calculate_rsi(self, prices: List[float], period: int, smoothing: int = 1) -> float:
        """حساب RSI"""
        if len(prices) < period + 1:
            return 50.0
        
        deltas = [prices[i] - prices[i-1] for i in range(1, len(prices))]
        gains = [d if d > 0 else 0 for d in deltas[-period:]]
        losses = [-d if d < 0 else 0 for d in deltas[-period:]]
        
        avg_gain = sum(gains) / period
        avg_loss = sum(losses) / period
        
        if avg_loss == 0:
            return 100.0
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def calculate_wt(self, hlc3: List[float], channel_length: int, average_length: int) -> float:
        """حساب WaveTrend"""
        if len(hlc3) < channel_length + average_length:
            return 0.0
        
        esa = sum(hlc3[-channel_length:]) / channel_length
        deviation = sum([abs(p - esa) for p in hlc3[-channel_length:]]) / channel_length
        
        if deviation == 0:
            ci = 0
        else:
            ci = (hlc3[-1] - esa) / (0.015 * deviation)
        
        wt = sum([ci] * average_length) / average_length
        return wt
    
    def calculate_cci(self, prices: List[float], period: int, smoothing: int = 1) -> float:
        """حساب CCI"""
        if len(prices) < period:
            return 0.0
        
        tp = prices[-1]
        sma = sum(prices[-period:]) / period
        mean_deviation = sum([abs(p - sma) for p in prices[-period:]]) / period
        
        if mean_deviation == 0:
            return 0.0
        
        cci = (tp - sma) / (0.015 * mean_deviation)
        return cci
    
    def calculate_adx(self, highs: List[float], lows: List[float], closes: List[float], period: int) -> float:
        """حساب ADX"""
        if len(highs) < period + 1:
            return 25.0
        
        tr_list = []
        plus_dm_list = []
        minus_dm_list = []
        
        for i in range(1, min(period + 1, len(highs))):
            tr = max(highs[-i] - lows[-i], 
                     abs(highs[-i] - closes[-i-1]), 
                     abs(lows[-i] - closes[-i-1]))
            tr_list.append(tr)
            
            plus_dm = highs[-i] - highs[-i-1] if highs[-i] - highs[-i-1] > lows[-i-1] - lows[-i] else 0
            minus_dm = lows[-i-1] - lows[-i] if lows[-i-1] - lows[-i] > highs[-i] - highs[-i-1] else 0
            
            plus_dm_list.append(plus_dm if plus_dm > minus_dm else 0)
            minus_dm_list.append(minus_dm if minus_dm > plus_dm else 0)
        
        atr = sum(tr_list) / len(tr_list) if tr_list else 1
        plus_di = 100 * sum(plus_dm_list) / (len(plus_dm_list) * atr) if atr > 0 else 0
        minus_di = 100 * sum(minus_dm_list) / (len(minus_dm_list) * atr) if atr > 0 else 0
        
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di) if (plus_di + minus_di) > 0 else 0
        return dx
    
    def calculate_feature(self, feature_type: str, param_a: int, param_b: int) -> float:
        """حساب قيمة الميزة"""
        closes = [d['close'] for d in self.price_data]
        highs = [d['high'] for d in self.price_data]
        lows = [d['low'] for d in self.price_data]
        hlc3 = [(d['high'] + d['low'] + d['close']) / 3 for d in self.price_data]
        
        if feature_type == "RSI":
            return self.calculate_rsi(closes, param_a, param_b)
        elif feature_type == "WT":
            return self.calculate_wt(hlc3, param_a, param_b)
        elif feature_type == "CCI":
            return self.calculate_cci(closes, param_a, param_b)
        elif feature_type == "ADX":
            return self.calculate_adx(highs, lows, closes, param_a)
        return 0.0
    
    def calculate_features(self) -> Dict[str, float]:
        """حساب جميع الميزات"""
        features = {}
        for key, params in self.feature_params.items():
            features[key] = self.calculate_feature(
                params['type'], params['paramA'], params['paramB']
            )
        return features
    
    def lorentzian_distance(self, features: Dict[str, float], index: int) -> float:
        """حساب المسافة اللورنتزية"""
        distance = 0.0
        for i, key in enumerate(['f1', 'f2', 'f3', 'f4', 'f5'][:self.feature_count]):
            if index < len(self.feature_arrays[key]):
                distance += np.log(1 + abs(features[key] - self.feature_arrays[key][index]))
        return distance
    
    def volatility_filter(self) -> bool:
        """فلتر التقلب"""
        if not self.use_volatility_filter or len(self.price_data) < 10:
            return True
        
        closes = [d['close'] for d in self.price_data[-10:]]
        returns = [abs(closes[i] - closes[i-1]) / closes[i-1] * 100 
                   for i in range(1, len(closes))]
        avg_volatility = sum(returns) / len(returns)
        return avg_volatility > 0.1
    
    def regime_filter(self) -> bool:
        """فلتر النظام"""
        if not self.use_regime_filter or len(self.price_data) < 50:
            return True
        
        ohlc4 = [(d['open'] + d['high'] + d['low'] + d['close']) / 4 
                 for d in self.price_data[-50:]]
        
        # Simple trend detection
        short_ma = sum(ohlc4[-10:]) / 10
        long_ma = sum(ohlc4) / len(ohlc4)
        
        trend = (short_ma - long_ma) / long_ma * 100
        return trend > self.regime_threshold
    
    def adx_filter(self) -> bool:
        """فلتر ADX"""
        if not self.use_adx_filter or len(self.price_data) < 14:
            return True
        
        highs = [d['high'] for d in self.price_data[-14:]]
        lows = [d['low'] for d in self.price_data[-14:]]
        closes = [d['close'] for d in self.price_data[-14:]]
        
        adx = self.calculate_adx(highs, lows, closes, 14)
        return adx > self.adx_threshold
    
    def update(self, candle: Dict[str, float]) -> int:
        """
        تحديث المصنف ببيانات شمعة جديدة وإرجاع التنبؤ
        Returns: 1 (شراء), -1 (بيع), 0 (محايد)
        """
        self.price_data.append(candle)
        
        # Keep only necessary history
        if len(self.price_data) > self.max_bars_back:
            self.price_data = self.price_data[-self.max_bars_back:]
        
        # Calculate features
        features = self.calculate_features()
        
        # Update feature arrays
        for key in ['f1', 'f2', 'f3', 'f4', 'f5']:
            self.feature_arrays[key].append(features[key])
            if len(self.feature_arrays[key]) > self.max_bars_back:
                self.feature_arrays[key] = self.feature_arrays[key][-self.max_bars_back:]
        
        # Calculate training label (price direction 4 bars ahead)
        if len(self.price_data) >= 5:
            price_4_bars_ago = self.price_data[-5]['close']
            current_price = self.price_data[-1]['close']
            
            if current_price > price_4_bars_ago:
                label = 1  # Long
            elif current_price < price_4_bars_ago:
                label = -1  # Short
            else:
                label = 0  # Neutral
            
            self.y_train.append(label)
            if len(self.y_train) > self.max_bars_back:
                self.y_train = self.y_train[-self.max_bars_back:]
        
        # Need enough data for prediction
        if len(self.y_train) < self.neighbors_count * 4:
            return 0
        
        # Apply filters
        if not (self.volatility_filter() and self.regime_filter() and self.adx_filter()):
            return 0
        
        # Approximate Nearest Neighbors with Lorentzian Distance
        distances = []
        predictions = []
        last_distance = -1.0
        
        size = min(self.max_bars_back - 1, len(self.y_train) - 1)
        
        for i in range(0, size, 4):  # Step by 4 bars
            d = self.lorentzian_distance(features, i)
            
            if d >= last_distance:
                last_distance = d
                distances.append(d)
                predictions.append(round(self.y_train[i]))
                
                if len(predictions) > self.neighbors_count:
                    # Reset last_distance to 75th percentile
                    sorted_distances = sorted(distances)
                    idx = int(self.neighbors_count * 3 / 4)
                    last_distance = sorted_distances[idx] if idx < len(sorted_distances) else distances[0]
                    distances.pop(0)
                    predictions.pop(0)
        
        # Sum predictions for final signal
        if predictions:
            prediction_sum = sum(predictions)
            if prediction_sum > 0:
                return 1  # Buy signal
            elif prediction_sum < 0:
                return -1  # Sell signal
        
        return 0


# ==============================================================================
# cTrader Open API Client
# ==============================================================================

class CTraderAPI:
    """
    عميل cTrader Open API للتداول
    """
    
    def __init__(self, settings: Dict[str, str]):
        self.client_id = settings.get("client_id", "")
        self.client_secret = settings.get("client_secret", "")
        self.access_token = settings.get("access_token", "")
        self.refresh_token = settings.get("refresh_token", "")
        self.account_id = settings.get("account_id", "")
        self.environment = settings.get("environment", "demo")
        self.api_url = settings.get("api_url", "https://demo.ctraderapi.com:5035")
        
        self.is_connected = False
        self.ws = None
        self.message_callbacks: Dict[str, Callable] = {}
        self.pending_requests: Dict[str, asyncio.Future] = {}
        self.request_id = 0
        
    def _get_request_id(self) -> int:
        """الحصول على معرف طلب فريد"""
        self.request_id += 1
        return self.request_id
    
    def connect(self) -> bool:
        """الاتصال بـ cTrader API"""
        try:
            if not all([self.client_id, self.client_secret, self.access_token]):
                logger.error("Missing cTrader API credentials. Please fill in CTRADER_SETTINGS.")
                return False
            
            # For REST API
            headers = {
                "Authorization": f"Bearer {self.access_token}",
                "Content-Type": "application/json"
            }
            
            # Test connection
            response = requests.get(
                f"{self.api_url}/v1/accounts",
                headers=headers,
                timeout=10
            )
            
            if response.status_code == 200:
                self.is_connected = True
                logger.info("Successfully connected to cTrader API")
                return True
            else:
                logger.error(f"Failed to connect: {response.status_code} - {response.text}")
                return False
                
        except Exception as e:
            logger.error(f"Connection error: {e}")
            return False
    
    def get_account_info(self) -> Optional[Dict]:
        """الحصول على معلومات الحساب"""
        if not self.is_connected:
            return None
        
        try:
            headers = {
                "Authorization": f"Bearer {self.access_token}",
                "Content-Type": "application/json"
            }
            
            response = requests.get(
                f"{self.api_url}/v1/accounts/{self.account_id}",
                headers=headers,
                timeout=10
            )
            
            if response.status_code == 200:
                return response.json()
            return None
            
        except Exception as e:
            logger.error(f"Error getting account info: {e}")
            return None
    
    def get_symbol_info(self, symbol: str) -> Optional[Dict]:
        """الحصول على معلومات الرمز"""
        if not self.is_connected:
            return None
        
        try:
            headers = {
                "Authorization": f"Bearer {self.access_token}",
                "Content-Type": "application/json"
            }
            
            response = requests.get(
                f"{self.api_url}/v1/symbols/{symbol}",
                headers=headers,
                timeout=10
            )
            
            if response.status_code == 200:
                return response.json()
            return None
            
        except Exception as e:
            logger.error(f"Error getting symbol info: {e}")
            return None
    
    def open_position(
        self,
        symbol: str,
        position_type: PositionType,
        volume: float,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None
    ) -> Optional[str]:
        """
        فتح مركز جديد
        
        Args:
            symbol: رمز الأداة المالية
            position_type: نوع المركز (شراء/بيع)
            volume: حجم الصفقة باللوت
            stop_loss: سعر وقف الخسارة (اختياري)
            take_profit: سعر جني الأرباح (اختياري)
        
        Returns:
            معرف الصفقة أو None في حالة الفشل
        """
        if not self.is_connected:
            logger.error("Not connected to cTrader API")
            return None
        
        try:
            headers = {
                "Authorization": f"Bearer {self.access_token}",
                "Content-Type": "application/json"
            }
            
            trade_side = "BUY" if position_type == PositionType.LONG else "SELL"
            
            payload = {
                "accountId": self.account_id,
                "symbol": symbol,
                "side": trade_side,
                "volume": volume,
                "orderType": "MARKET"
            }
            
            if stop_loss:
                payload["stopLoss"] = stop_loss
            if take_profit:
                payload["takeProfit"] = take_profit
            
            response = requests.post(
                f"{self.api_url}/v1/orders",
                headers=headers,
                json=payload,
                timeout=10
            )
            
            if response.status_code in [200, 201]:
                result = response.json()
                trade_id = result.get("orderId", result.get("positionId", str(int(time.time()))))
                logger.info(f"Position opened: {trade_id} - {position_type.value} {volume} lots {symbol}")
                return trade_id
            else:
                logger.error(f"Failed to open position: {response.status_code} - {response.text}")
                return None
                
        except Exception as e:
            logger.error(f"Error opening position: {e}")
            return None
    
    def close_position(self, position_id: str) -> bool:
        """إغلاق مركز"""
        if not self.is_connected:
            logger.error("Not connected to cTrader API")
            return False
        
        try:
            headers = {
                "Authorization": f"Bearer {self.access_token}",
                "Content-Type": "application/json"
            }
            
            response = requests.delete(
                f"{self.api_url}/v1/positions/{position_id}",
                headers=headers,
                timeout=10
            )
            
            if response.status_code in [200, 204]:
                logger.info(f"Position closed: {position_id}")
                return True
            else:
                logger.error(f"Failed to close position: {response.status_code} - {response.text}")
                return False
                
        except Exception as e:
            logger.error(f"Error closing position: {e}")
            return False
    
    def get_positions(self) -> List[Dict]:
        """الحصول على قائمة المراكز المفتوحة"""
        if not self.is_connected:
            return []
        
        try:
            headers = {
                "Authorization": f"Bearer {self.access_token}",
                "Content-Type": "application/json"
            }
            
            response = requests.get(
                f"{self.api_url}/v1/accounts/{self.account_id}/positions",
                headers=headers,
                timeout=10
            )
            
            if response.status_code == 200:
                return response.json().get("positions", [])
            return []
            
        except Exception as e:
            logger.error(f"Error getting positions: {e}")
            return []


# ==============================================================================
# TradingView WebSocket Data Feed
# ==============================================================================

class TradingViewDataFeed:
    """
    مصدر بيانات TradingView عبر WebSocket
    """
    
    def __init__(self, symbol: str = "EURUSD", timeframe: str = "1m"):
        self.symbol = symbol
        self.timeframe = timeframe
        self.ws_url = "wss://data.tradingview.com/socket.io/websocket"
        self.ws = None
        self.is_connected = False
        self.on_candle_callback: Optional[Callable[[Dict], None]] = None
        self.candles: List[Dict[str, float]] = []
        
    def _generate_session_id(self) -> str:
        """إنشاء معرف جلسة عشوائي"""
        import random
        import string
        return "qs_" + "".join(random.choices(string.ascii_lowercase + string.digits, k=12))
    
    def _create_message(self, payload: Dict) -> str:
        """إنشاء رسالة WebSocket"""
        message = json.dumps(payload)
        return f"~m~{len(message)}~m~{message}"
    
    def connect(self) -> bool:
        """الاتصال بـ TradingView WebSocket"""
        try:
            self.ws = websocket.WebSocketApp(
                self.ws_url,
                on_open=self._on_open,
                on_message=self._on_message,
                on_error=self._on_error,
                on_close=self._on_close
            )
            
            # Start WebSocket in a separate thread
            self.ws_thread = threading.Thread(target=self.ws.run_forever, kwargs={
                'sslopt': {"cert_reqs": ssl.CERT_NONE}
            })
            self.ws_thread.daemon = True
            self.ws_thread.start()
            
            # Wait for connection
            timeout = 10
            start_time = time.time()
            while not self.is_connected and time.time() - start_time < timeout:
                time.sleep(0.1)
            
            return self.is_connected
            
        except Exception as e:
            logger.error(f"Error connecting to TradingView: {e}")
            return False
    
    def _on_open(self, ws):
        """معالجة فتح الاتصال"""
        logger.info("TradingView WebSocket connected")
        self.is_connected = True
        
        # Send chart create session
        session_id = self._generate_session_id()
        
        # Create chart session
        ws.send(self._create_message({
            "m": "chart_create_session",
            "p": [session_id, ""]
        }))
        
        # Resolve symbol
        ws.send(self._create_message({
            "m": "resolve_symbol",
            "p": [session_id, f"={self.symbol}"]
        }))
        
        # Create series
        ws.send(self._create_message({
            "m": "create_series",
            "p": [session_id, "s1", f"={self.symbol}", self.timeframe, 300]
        }))
    
    def _on_message(self, ws, message):
        """معالجة الرسائل الواردة"""
        try:
            # Parse message
            if message.startswith("~m~"):
                parts = message.split("~m~")
                if len(parts) >= 3:
                    msg_len = int(parts[1])
                    msg_data = parts[2][:msg_len]
                    
                    try:
                        data = json.loads(msg_data)
                        
                        # Handle candle data
                        if data.get("m") == "timescale_update":
                            self._process_candles(data.get("p", []))
                        elif data.get("m") == "du":
                            self._process_candles(data.get("p", []))
                            
                    except json.JSONDecodeError:
                        pass
                        
        except Exception as e:
            logger.error(f"Error processing message: {e}")
    
    def _process_candles(self, data):
        """معالجة بيانات الشموع"""
        try:
            if isinstance(data, list) and len(data) > 1:
                series_data = data[1]
                if "s" in series_data and series_data["s"]:
                    for candle in series_data["s"]:
                        candle_data = {
                            "timestamp": candle.get("t"),
                            "open": float(candle.get("o", 0)),
                            "high": float(candle.get("h", 0)),
                            "low": float(candle.get("l", 0)),
                            "close": float(candle.get("c", 0)),
                            "volume": float(candle.get("v", 0))
                        }
                        
                        self.candles.append(candle_data)
                        
                        # Keep only recent candles
                        if len(self.candles) > 5000:
                            self.candles = self.candles[-5000:]
                        
                        # Notify callback
                        if self.on_candle_callback:
                            self.on_candle_callback(candle_data)
                            
        except Exception as e:
            logger.error(f"Error processing candles: {e}")
    
    def _on_error(self, ws, error):
        """معالجة الأخطاء"""
        logger.error(f"TradingView WebSocket error: {error}")
    
    def _on_close(self, ws, close_status_code, close_msg):
        """معالجة إغلاق الاتصال"""
        logger.info("TradingView WebSocket disconnected")
        self.is_connected = False
    
    def disconnect(self):
        """قطع الاتصال"""
        if self.ws:
            self.ws.close()
        self.is_connected = False
    
    def set_on_candle_callback(self, callback: Callable[[Dict], None]):
        """تعيين دالة استدعاء عند استلام شمعة جديدة"""
        self.on_candle_callback = callback


# ==============================================================================
# Main Trading Bot
# ==============================================================================

class LorentzianTradingBot:
    """
    روبوت التداول الرئيسي
    """
    
    def __init__(
        self,
        ctrader_settings: Dict[str, str],
        trading_settings: Dict[str, Any]
    ):
        self.ctrader_settings = ctrader_settings
        self.trading_settings = trading_settings
        
        # Initialize components
        self.classifier = LorentzianClassifier(
            neighbors_count=8,
            max_bars_back=2000,
            feature_count=5
        )
        
        self.ctrader = CTraderAPI(ctrader_settings)
        self.data_feed = TradingViewDataFeed(
            symbol=trading_settings.get("symbol", "EURUSD"),
            timeframe=trading_settings.get("timeframe", "1m")
        )
        
        # Account state
        self.account = AccountState(
            balance=trading_settings.get("initial_capital", 100.0),
            leverage=trading_settings.get("leverage", 10000)
        )
        
        # Trading state
        self.current_trade: Optional[Trade] = None
        self.is_running = False
        self.last_signal = SignalType.NEUTRAL
        
        # Statistics
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        
    def start(self) -> bool:
        """بدء الروبوت"""
        logger.info("=" * 60)
        logger.info("Starting Lorentzian Classification Trading Bot")
        logger.info("=" * 60)
        
        # Check credentials
        if not self.ctrader_settings.get("client_id"):
            logger.error("=" * 60)
            logger.error("cTrader API credentials are missing!")
            logger.error("Please fill in the CTRADER_SETTINGS dictionary:")
            logger.error("  - client_id")
            logger.error("  - client_secret")
            logger.error("  - access_token")
            logger.error("  - account_id")
            logger.error("=" * 60)
            return False
        
        # Connect to cTrader
        logger.info("Connecting to cTrader API...")
        if not self.ctrader.connect():
            logger.error("Failed to connect to cTrader API")
            return False
        
        # Get account info
        account_info = self.ctrader.get_account_info()
        if account_info:
            logger.info(f"Account Info: {account_info}")
        
        # Set up data feed callback
        self.data_feed.set_on_candle_callback(self._on_new_candle)
        
        # Connect to TradingView
        logger.info("Connecting to TradingView data feed...")
        if not self.data_feed.connect():
            logger.warning("Failed to connect to TradingView, using fallback data source")
        
        self.is_running = True
        logger.info("Bot started successfully!")
        logger.info(f"Symbol: {self.trading_settings['symbol']}")
        logger.info(f"Initial Capital: ${self.account.balance:.2f}")
        logger.info(f"Leverage: 1:{self.account.leverage}")
        logger.info("=" * 60)
        
        return True
    
    def stop(self):
        """إيقاف الروبوت"""
        logger.info("Stopping bot...")
        self.is_running = False
        
        # Close any open position
        if self.current_trade:
            self._close_current_trade()
        
        # Disconnect
        self.data_feed.disconnect()
        
        # Print statistics
        self._print_statistics()
        
        logger.info("Bot stopped")
    
    def _on_new_candle(self, candle: Dict[str, float]):
        """معالجة شمعة جديدة"""
        try:
            # Update classifier
            signal_value = self.classifier.update(candle)
            
            # Convert to SignalType
            if signal_value == 1:
                signal = SignalType.BUY
            elif signal_value == -1:
                signal = SignalType.SELL
            else:
                signal = SignalType.NEUTRAL
            
            # Process signal
            self._process_signal(signal, candle)
            
        except Exception as e:
            logger.error(f"Error processing candle: {e}")
    
    def _process_signal(self, signal: SignalType, candle: Dict[str, float]):
        """معالجة الإشارة"""
        current_price = candle["close"]
        
        # Log signal
        if signal != self.last_signal:
            logger.info(f"Signal: {signal.name} at price {current_price}")
            self.last_signal = signal
        
        # Execute trades based on signal
        if signal == SignalType.BUY:
            if self.account.current_position == PositionType.SHORT:
                # Close short and open long
                self._close_current_trade(current_price)
                self._open_long_position(current_price)
            elif self.account.current_position == PositionType.NONE:
                # Open long position
                self._open_long_position(current_price)
                
        elif signal == SignalType.SELL:
            if self.account.current_position == PositionType.LONG:
                # Close long and open short
                self._close_current_trade(current_price)
                self._open_short_position(current_price)
            elif self.account.current_position == PositionType.NONE:
                # Open short position
                self._open_short_position(current_price)
    
    def _open_long_position(self, price: float):
        """فتح مركز شراء"""
        try:
            # Calculate volume with leverage
            volume = self._calculate_volume()
            
            # Open position via cTrader
            trade_id = self.ctrader.open_position(
                symbol=self.trading_settings["symbol"],
                position_type=PositionType.LONG,
                volume=volume
            )
            
            if trade_id:
                # Create trade record
                trade = Trade(
                    id=trade_id,
                    symbol=self.trading_settings["symbol"],
                    position_type=PositionType.LONG,
                    entry_price=price,
                    volume=volume,
                    entry_time=datetime.now()
                )
                
                self.current_trade = trade
                self.account.open_trades.append(trade)
                self.account.current_position = PositionType.LONG
                
                logger.info(f"OPENED LONG: {volume} lots at {price}")
                logger.info(f"Balance: ${self.account.balance:.2f}")
                
        except Exception as e:
            logger.error(f"Error opening long position: {e}")
    
    def _open_short_position(self, price: float):
        """فتح مركز بيع"""
        try:
            # Calculate volume with leverage
            volume = self._calculate_volume()
            
            # Open position via cTrader
            trade_id = self.ctrader.open_position(
                symbol=self.trading_settings["symbol"],
                position_type=PositionType.SHORT,
                volume=volume
            )
            
            if trade_id:
                # Create trade record
                trade = Trade(
                    id=trade_id,
                    symbol=self.trading_settings["symbol"],
                    position_type=PositionType.SHORT,
                    entry_price=price,
                    volume=volume,
                    entry_time=datetime.now()
                )
                
                self.current_trade = trade
                self.account.open_trades.append(trade)
                self.account.current_position = PositionType.SHORT
                
                logger.info(f"OPENED SHORT: {volume} lots at {price}")
                logger.info(f"Balance: ${self.account.balance:.2f}")
                
        except Exception as e:
            logger.error(f"Error opening short position: {e}")
    
    def _close_current_trade(self, exit_price: Optional[float] = None):
        """إغلاق المركز الحالي"""
        if not self.current_trade:
            return
        
        try:
            trade = self.current_trade
            
            # Get exit price if not provided
            if exit_price is None:
                # Try to get current price from cTrader
                symbol_info = self.ctrader.get_symbol_info(trade.symbol)
                if symbol_info:
                    exit_price = symbol_info.get("bid", trade.entry_price)
                else:
                    exit_price = trade.entry_price
            
            # Calculate profit/loss
            if trade.position_type == PositionType.LONG:
                pips = (exit_price - trade.entry_price) / 0.0001  # For 5-digit brokers
                profit = pips * trade.volume * 10  # Simplified P&L calculation
            else:
                pips = (trade.entry_price - exit_price) / 0.0001
                profit = pips * trade.volume * 10
            
            # Apply leverage
            profit *= self.account.leverage / 100
            
            # Update trade record
            trade.exit_price = exit_price
            trade.exit_time = datetime.now()
            trade.profit_loss = profit
            trade.is_closed = True
            
            # Close position via cTrader
            self.ctrader.close_position(trade.id)
            
            # Update account
            self.account.balance += profit
            self.account.cumulative_profit += profit
            self.account.closed_trades.append(trade)
            self.account.current_position = PositionType.NONE
            
            # Update statistics
            self.total_trades += 1
            if profit > 0:
                self.winning_trades += 1
            else:
                self.losing_trades += 1
            
            # Log
            profit_str = f"+${profit:.2f}" if profit > 0 else f"-${abs(profit):.2f}"
            logger.info(f"CLOSED {trade.position_type.value}: {profit_str} at {exit_price}")
            logger.info(f"New Balance: ${self.account.balance:.2f}")
            
            self.current_trade = None
            
        except Exception as e:
            logger.error(f"Error closing trade: {e}")
    
    def _calculate_volume(self) -> float:
        """حساب حجم الصفقة مع التراكم"""
        if self.trading_settings.get("use_compound", True):
            # Use current balance for compounding
            base_volume = self.trading_settings.get("volume_lots", 0.01)
            
            # Scale volume based on balance growth
            balance_ratio = self.account.balance / self.trading_settings.get("initial_capital", 100.0)
            volume = base_volume * max(1.0, balance_ratio * 0.5)  # Conservative scaling
            
            # Cap maximum volume
            return min(volume, 10.0)
        else:
            return self.trading_settings.get("volume_lots", 0.01)
    
    def _print_statistics(self):
        """طباعة إحصائيات التداول"""
        logger.info("=" * 60)
        logger.info("TRADING STATISTICS")
        logger.info("=" * 60)
        logger.info(f"Total Trades: {self.total_trades}")
        logger.info(f"Winning Trades: {self.winning_trades}")
        logger.info(f"Losing Trades: {self.losing_trades}")
        
        if self.total_trades > 0:
            win_rate = (self.winning_trades / self.total_trades) * 100
            logger.info(f"Win Rate: {win_rate:.2f}%")
        
        logger.info(f"Final Balance: ${self.account.balance:.2f}")
        logger.info(f"Total Profit: ${self.account.cumulative_profit:.2f}")
        logger.info(f"Return: {(self.account.cumulative_profit / self.trading_settings.get('initial_capital', 100.0)) * 100:.2f}%")
        logger.info("=" * 60)
    
    def run(self):
        """تشغيل الروبوت"""
        if not self.start():
            return
        
        try:
            # Keep the bot running
            while self.is_running:
                time.sleep(1)
                
                # Update account info periodically
                if self.total_trades % 10 == 0:
                    account_info = self.ctrader.get_account_info()
                    if account_info:
                        logger.debug(f"Account update: {account_info}")
                        
        except KeyboardInterrupt:
            logger.info("Keyboard interrupt received")
        finally:
            self.stop()


# ==============================================================================
# Alternative: Direct TradingView Alert Webhook
# ==============================================================================

class TradingViewWebhookServer:
    """
    خادم Webhook لاستقبال تنبيهات TradingView
    """
    
    def __init__(
        self,
        ctrader_settings: Dict[str, str],
        trading_settings: Dict[str, Any],
        host: str = "0.0.0.0",
        port: int = 5000
    ):
        self.ctrader_settings = ctrader_settings
        self.trading_settings = trading_settings
        self.host = host
        self.port = port
        
        self.ctrader = CTraderAPI(ctrader_settings)
        self.account = AccountState(
            balance=trading_settings.get("initial_capital", 100.0),
            leverage=trading_settings.get("leverage", 10000)
        )
        self.current_position = PositionType.NONE
        
    def start(self):
        """بدء الخادم"""
        from flask import Flask, request, jsonify
        
        app = Flask(__name__)
        
        @app.route('/webhook', methods=['POST'])
        def webhook():
            """استقبال تنبيهات TradingView"""
            try:
                data = request.get_json()
                logger.info(f"Received webhook: {data}")
                
                # Parse alert message
                message = data.get("message", "")
                
                if "Open Long" in message or "BUY" in message.upper():
                    self._execute_buy()
                elif "Open Short" in message or "SELL" in message.upper():
                    self._execute_sell()
                elif "Close Long" in message:
                    self._close_position()
                elif "Close Short" in message:
                    self._close_position()
                
                return jsonify({"status": "success"}), 200
                
            except Exception as e:
                logger.error(f"Webhook error: {e}")
                return jsonify({"status": "error", "message": str(e)}), 500
        
        @app.route('/health', methods=['GET'])
        def health():
            """فحص الصحة"""
            return jsonify({"status": "healthy"}), 200
        
        # Connect to cTrader
        if not self.ctrader.connect():
            logger.error("Failed to connect to cTrader API")
            return
        
        logger.info(f"Starting webhook server on {self.host}:{self.port}")
        app.run(host=self.host, port=self.port, debug=False)
    
    def _execute_buy(self):
        """تنفيذ أمر شراء"""
        if self.current_position == PositionType.SHORT:
            self._close_position()
        
        if self.current_position == PositionType.NONE:
            volume = self.trading_settings.get("volume_lots", 0.01)
            trade_id = self.ctrader.open_position(
                self.trading_settings["symbol"],
                PositionType.LONG,
                volume
            )
            if trade_id:
                self.current_position = PositionType.LONG
                logger.info(f"Webhook: Opened LONG position")
    
    def _execute_sell(self):
        """تنفيذ أمر بيع"""
        if self.current_position == PositionType.LONG:
            self._close_position()
        
        if self.current_position == PositionType.NONE:
            volume = self.trading_settings.get("volume_lots", 0.01)
            trade_id = self.ctrader.open_position(
                self.trading_settings["symbol"],
                PositionType.SHORT,
                volume
            )
            if trade_id:
                self.current_position = PositionType.SHORT
                logger.info(f"Webhook: Opened SHORT position")
    
    def _close_position(self):
        """إغلاق المركز"""
        positions = self.ctrader.get_positions()
        for pos in positions:
            self.ctrader.close_position(pos.get("positionId"))
        self.current_position = PositionType.NONE
        logger.info(f"Webhook: Closed all positions")


# ==============================================================================
# Main Entry Point
# ==============================================================================

def main():
    """النقطة الرئيسية للبرنامج"""
    
    print("=" * 70)
    print("   Lorentzian Classification Hybrid Trading Bot")
    print("=" * 70)
    print()
    print("IMPORTANT: Please fill in your cTrader API credentials in")
    print("the CTRADER_SETTINGS dictionary at the top of this file.")
    print()
    print("Required credentials:")
    print("  - client_id")
    print("  - client_secret")
    print("  - access_token")
    print("  - account_id")
    print()
    print("=" * 70)
    print()
    
    # Check if credentials are provided
    if not CTRADER_SETTINGS.get("client_id"):
        print("ERROR: cTrader API credentials are missing!")
        print("Please edit the CTRADER_SETTINGS at the top of this file.")
        print()
        print("To get cTrader API credentials:")
        print("1. Go to https://ctrader.com/")
        print("2. Log in to your account")
        print("3. Go to Settings > API Access")
        print("4. Create a new API application")
        print("5. Copy the Client ID and Client Secret")
        print("6. Generate an Access Token")
        print()
        return
    
    # Create and run bot
    bot = LorentzianTradingBot(CTRADER_SETTINGS, TRADING_SETTINGS)
    
    try:
        bot.run()
    except Exception as e:
        logger.error(f"Bot error: {e}")
        raise


def run_webhook_server():
    """تشغيل خادم Webhook"""
    server = TradingViewWebhookServer(CTRADER_SETTINGS, TRADING_SETTINGS)
    server.start()


if __name__ == "__main__":
    # Uncomment the mode you want to use:
    
    # Mode 1: Direct Trading (uses TradingView data feed)
    main()
    
    # Mode 2: Webhook Server (receives alerts from TradingView)
    # run_webhook_server()
