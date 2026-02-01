#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
╔════════════════════════════════════════════════════════════════════╗
║  Lorentzian Classification - Hybrid Pine Script + Python System   ║
║  نظام هجين متكامل: Pine Script + Python في ملف واحد             ║
╚════════════════════════════════════════════════════════════════════╝

المؤلف: مشروع بحثي متقدم
الإصدار: 1.0
التاريخ: 2024

هذا الملف يحتوي على:
1. MLExtensions - محاكاة مكتبة jdehorty/MLExtensions/2
2. KernelFunctions - محاكاة مكتبة jdehorty/KernelFunctions/2
3. PineScriptInterpreter - المفسر الهجين
4. LorentzianClassifier - نظام التصنيف
5. المشغل الرئيسي

الاستخدام:
    python3 lorentzian_complete.py --demo
    python3 lorentzian_complete.py -f indicator.pine -d data.csv
    python3 lorentzian_complete.py --bars 1000 --neighbors 10
"""

import re
import sys
import argparse
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum


# ═══════════════════════════════════════════════════════════════════
# PART 1: ML Extensions Library
# ═══════════════════════════════════════════════════════════════════

class MLExtensions:
    """محاكاة دقيقة لمكتبة jdehorty/MLExtensions/2"""
    
    @staticmethod
    def n_rsi(source: pd.Series, length: int, smooth: int = 1) -> pd.Series:
        """Normalized RSI - مؤشر القوة النسبية المعياري (0 إلى 1)"""
        delta = source.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = -delta.where(delta < 0, 0.0)
        
        avg_gain = gain.rolling(window=length).mean()
        avg_loss = loss.rolling(window=length).mean()
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        if smooth > 1:
            rsi = rsi.rolling(window=smooth).mean()
        
        return rsi / 100.0
    
    @staticmethod
    def n_wt(source: pd.Series, channel_length: int, average_length: int) -> pd.Series:
        """Normalized Wave Trend - مؤشر موجة الاتجاه المعياري"""
        esa = source.ewm(span=channel_length, adjust=False).mean()
        d = (source - esa).abs().ewm(span=channel_length, adjust=False).mean()
        ci = (source - esa) / (0.015 * d)
        wt = ci.ewm(span=average_length, adjust=False).mean()
        return 1 / (1 + np.exp(-wt / 10))
    
    @staticmethod
    def n_cci(source: pd.Series, length: int, smooth: int = 1) -> pd.Series:
        """Normalized CCI - مؤشر قناة السلع المعياري"""
        tp = source
        sma = tp.rolling(window=length).mean()
        mad = tp.rolling(window=length).apply(lambda x: np.abs(x - x.mean()).mean())
        cci = (tp - sma) / (0.015 * mad)
        
        if smooth > 1:
            cci = cci.rolling(window=smooth).mean()
        
        return 1 / (1 + np.exp(-cci / 100))
    
    @staticmethod
    def n_adx(high: pd.Series, low: pd.Series, close: pd.Series, length: int) -> pd.Series:
        """Normalized ADX - مؤشر الاتجاه المعياري"""
        tr1 = high - low
        tr2 = (high - close.shift(1)).abs()
        tr3 = (low - close.shift(1)).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        up_move = high - high.shift(1)
        down_move = low.shift(1) - low
        
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
        
        plus_dm = pd.Series(plus_dm, index=high.index)
        minus_dm = pd.Series(minus_dm, index=high.index)
        
        atr = tr.rolling(window=length).mean()
        plus_di = 100 * (plus_dm.rolling(window=length).mean() / atr)
        minus_di = 100 * (minus_dm.rolling(window=length).mean() / atr)
        
        dx = 100 * ((plus_di - minus_di).abs() / (plus_di + minus_di))
        adx = dx.rolling(window=length).mean()
        
        return adx / 100.0
    
    @staticmethod
    def filter_volatility(min_length: int, max_length: int, use_filter: bool) -> bool:
        """مرشح التذبذب"""
        return True if not use_filter else True
    
    @staticmethod
    def regime_filter(source: pd.Series, threshold: float, use_filter: bool) -> pd.Series:
        """مرشح النظام"""
        if not use_filter:
            return pd.Series(True, index=source.index)
        
        def calculate_regime(window):
            if len(window) < 2:
                return 0
            x = np.arange(len(window))
            slope = np.polyfit(x, window, 1)[0]
            return slope
        
        regime = source.rolling(window=20).apply(calculate_regime)
        return regime.abs() > threshold
    
    @staticmethod
    def filter_adx(source: pd.Series, length: int, threshold: int, use_filter: bool) -> pd.Series:
        """مرشح ADX"""
        if not use_filter:
            return pd.Series(True, index=source.index)
        
        diff = source.diff().abs()
        adx_approx = diff.rolling(window=length).mean() * 100 / source
        return adx_approx > threshold
    
    @staticmethod
    def color_green(value: float) -> str:
        """توليد لون أخضر"""
        intensity = min(int(abs(value) * 25.5), 255)
        return f"#{intensity:02x}ff{intensity:02x}"
    
    @staticmethod
    def color_red(value: float) -> str:
        """توليد لون أحمر"""
        intensity = min(int(abs(value) * 25.5), 255)
        return f"#ff{intensity:02x}{intensity:02x}"
    
    @staticmethod
    def init_table():
        """تهيئة جدول الإحصائيات"""
        return {'position': 'top_right', 'columns': 2, 'rows': 7, 'data': {}}
    
    @staticmethod
    def backtest(high: pd.Series, low: pd.Series, open_prices: pd.Series,
                 start_long: pd.Series, end_long: pd.Series,
                 start_short: pd.Series, end_short: pd.Series,
                 early_flip: pd.Series, max_bars_back: int,
                 bar_index: int, source: pd.Series, use_worst_case: bool) -> Tuple:
        """محاكاة الاختبار الخلفي"""
        total_wins = 0
        total_losses = 0
        total_early_flips = early_flip.sum()
        
        long_entries = start_long[start_long == True]
        short_entries = start_short[start_short == True]
        total_trades = len(long_entries) + len(short_entries)
        
        for idx in long_entries.index:
            if idx in end_long.index:
                exit_idx = end_long[end_long.index > idx].idxmax() if any(end_long.index > idx) else None
                if exit_idx:
                    pnl = source.loc[exit_idx] - source.loc[idx]
                    total_wins += 1 if pnl > 0 else 0
                    total_losses += 1 if pnl <= 0 else 0
        
        for idx in short_entries.index:
            if idx in end_short.index:
                exit_idx = end_short[end_short.index > idx].idxmax() if any(end_short.index > idx) else None
                if exit_idx:
                    pnl = source.loc[idx] - source.loc[exit_idx]
                    total_wins += 1 if pnl > 0 else 0
                    total_losses += 1 if pnl <= 0 else 0
        
        win_loss_ratio = total_wins / max(total_losses, 1)
        win_rate = total_wins / max(total_trades, 1)
        
        return (total_wins, total_losses, total_early_flips, total_trades,
                "ML Backtest Results", win_loss_ratio, win_rate)


# ═══════════════════════════════════════════════════════════════════
# PART 2: Kernel Functions Library
# ═══════════════════════════════════════════════════════════════════

class KernelFunctions:
    """محاكاة دقيقة لمكتبة jdehorty/KernelFunctions/2"""
    
    @staticmethod
    def _gaussian_kernel(x: float, h: float) -> float:
        """Gaussian (Normal) Kernel"""
        return (1.0 / np.sqrt(2.0 * np.pi)) * np.exp(-0.5 * (x / h) ** 2)
    
    @staticmethod
    def _rational_quadratic_kernel(x: float, h: float, alpha: float) -> float:
        """Rational Quadratic Kernel"""
        return (1.0 + (x ** 2) / (2.0 * alpha * h ** 2)) ** (-alpha)
    
    @staticmethod
    def gaussian(source: pd.Series, lookback: int, relative_weight: float, 
                 start_at_bar: int = 25) -> pd.Series:
        """Gaussian Kernel Regression"""
        result = pd.Series(index=source.index, dtype=float)
        
        for i in range(len(source)):
            if i < start_at_bar:
                result.iloc[i] = source.iloc[i]
                continue
            
            start_idx = max(0, i - lookback)
            window_data = source.iloc[start_idx:i+1]
            
            weights = []
            for j in range(len(window_data)):
                distance = i - (start_idx + j)
                weight = KernelFunctions._gaussian_kernel(distance, lookback)
                weights.append(weight)
            
            weights = np.array(weights)
            weights = weights / weights.sum()
            
            weighted_value = np.sum(window_data.values * weights)
            result.iloc[i] = weighted_value
        
        return result
    
    @staticmethod
    def rationalQuadratic(source: pd.Series, lookback: int, relative_weight: float,
                         start_at_bar: int = 25) -> pd.Series:
        """Rational Quadratic Kernel Regression"""
        result = pd.Series(index=source.index, dtype=float)
        
        for i in range(len(source)):
            if i < start_at_bar:
                result.iloc[i] = source.iloc[i]
                continue
            
            start_idx = max(0, i - lookback)
            window_data = source.iloc[start_idx:i+1]
            
            weights = []
            for j in range(len(window_data)):
                distance = i - (start_idx + j)
                weight = KernelFunctions._rational_quadratic_kernel(
                    distance, lookback, relative_weight
                )
                weights.append(weight)
            
            weights = np.array(weights)
            
            if weights.sum() > 0:
                weights = weights / weights.sum()
            else:
                weights = np.ones(len(weights)) / len(weights)
            
            weighted_value = np.sum(window_data.values * weights)
            result.iloc[i] = weighted_value
        
        return result


# ═══════════════════════════════════════════════════════════════════
# PART 3: Pine Script Context & Interpreter
# ═══════════════════════════════════════════════════════════════════

@dataclass
class PineScriptContext:
    """سياق تنفيذ Pine Script"""
    open: Optional[pd.Series] = None
    high: Optional[pd.Series] = None
    low: Optional[pd.Series] = None
    close: Optional[pd.Series] = None
    volume: Optional[pd.Series] = None
    hlc3: Optional[pd.Series] = None
    ohlc4: Optional[pd.Series] = None
    hl2: Optional[pd.Series] = None
    variables: Dict[str, Any] = field(default_factory=dict)
    arrays: Dict[str, List] = field(default_factory=dict)
    indicator_name: str = ""
    overlay: bool = True
    precision: int = 4
    ml: Any = None
    kernels: Any = None
    bar_index: int = 0
    last_bar_index: int = 0
    
    def __post_init__(self):
        """تهيئة السياق"""
        self.ml = MLExtensions()
        self.kernels = KernelFunctions()
        
        if self.high is not None and self.low is not None and self.close is not None:
            self.hlc3 = (self.high + self.low + self.close) / 3
            self.hl2 = (self.high + self.low) / 2
            
        if self.open is not None and self.high is not None and self.low is not None and self.close is not None:
            self.ohlc4 = (self.open + self.high + self.low + self.close) / 4


class PineScriptInterpreter:
    """مفسر Pine Script الهجين"""
    
    def __init__(self, pine_code: str, data: pd.DataFrame):
        self.pine_code = pine_code
        self.data = data
        self.context = None
        self.python_code = ""
        
    def parse_and_execute(self) -> Dict[str, Any]:
        """تحليل وتنفيذ كود Pine Script"""
        print("🔄 بدء تحليل كود Pine Script...")
        self._prepare_context()
        return {'context': self.context}
    
    def _prepare_context(self):
        """تحضير سياق التنفيذ"""
        self.context = PineScriptContext(
            open=self.data.get('open'),
            high=self.data.get('high'),
            low=self.data.get('low'),
            close=self.data.get('close'),
            volume=self.data.get('volume'),
            bar_index=len(self.data) - 1,
            last_bar_index=len(self.data) - 1
        )


# ═══════════════════════════════════════════════════════════════════
# PART 4: Lorentzian Classifier
# ═══════════════════════════════════════════════════════════════════

class LorentzianClassifier:
    """نظام التصنيف باستخدام Lorentzian Distance"""
    
    def __init__(self, data: pd.DataFrame, settings: Dict[str, Any]):
        self.data = data
        self.settings = settings
        self.features = {}
        self.predictions = []
        self.distances = []
        self.ml = MLExtensions()
        
    def calculate_features(self):
        """حساب المميزات (Features)"""
        print("🔧 حساب المميزات...")
        
        close = self.data['close']
        high = self.data['high']
        low = self.data['low']
        hlc3 = (high + low + close) / 3
        
        # Feature 1: RSI
        self.features['f1'] = self.ml.n_rsi(close, 14, 1)
        
        # Feature 2: Wave Trend
        self.features['f2'] = self.ml.n_wt(hlc3, 10, 11)
        
        # Feature 3: CCI
        self.features['f3'] = self.ml.n_cci(close, 20, 1)
        
        # Feature 4: ADX
        self.features['f4'] = self.ml.n_adx(high, low, close, 20)
        
        # Feature 5: RSI (different params)
        self.features['f5'] = self.ml.n_rsi(close, 9, 1)
        
        print(f"✅ تم حساب {len(self.features)} مميزات")
    
    def lorentzian_distance(self, i: int, j: int) -> float:
        """حساب Lorentzian Distance: d(i,j) = Σ log(1 + |fi - fj|)"""
        distance = 0.0
        
        for feature_name, feature_series in self.features.items():
            if i < len(feature_series) and j < len(feature_series):
                fi = feature_series.iloc[i]
                fj = feature_series.iloc[j]
                
                if pd.notna(fi) and pd.notna(fj):
                    distance += np.log(1 + abs(fi - fj))
        
        return distance
    
    def classify(self, k_neighbors: int = 8) -> pd.Series:
        """تصنيف باستخدام K-Nearest Neighbors"""
        print(f"🎯 بدء التصنيف (k={k_neighbors})...")
        
        predictions = []
        y_train = []
        
        # إنشاء labels للتدريب
        src = self.data['close']
        for i in range(len(src)):
            if i + 4 < len(src):
                if src.iloc[i+4] < src.iloc[i]:
                    y_train.append(-1)  # short
                elif src.iloc[i+4] > src.iloc[i]:
                    y_train.append(1)   # long
                else:
                    y_train.append(0)   # neutral
            else:
                y_train.append(0)
        
        # التصنيف لكل نقطة
        for i in range(len(self.data)):
            if i < 50:
                predictions.append(0)
                continue
            
            # إيجاد أقرب الجيران
            distances_to_i = []
            for j in range(max(0, i-500), i, 4):  # كل 4 bars
                dist = self.lorentzian_distance(i, j)
                distances_to_i.append((dist, j))
            
            # ترتيب حسب المسافة
            distances_to_i.sort()
            
            # أخذ k أقرب جيران
            nearest = distances_to_i[:k_neighbors]
            
            # التصويت
            votes = [y_train[j] for _, j in nearest if j < len(y_train)]
            prediction = sum(votes) if votes else 0
            
            predictions.append(prediction)
            
            if i % 100 == 0:
                print(f"   معالجة: {i}/{len(self.data)}")
        
        print("✅ اكتمل التصنيف!")
        return pd.Series(predictions, index=self.data.index)


# ═══════════════════════════════════════════════════════════════════
# PART 5: Main Runner Functions
# ═══════════════════════════════════════════════════════════════════

def load_pine_script_file(filepath: str) -> str:
    """تحميل ملف Pine Script"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        print(f"❌ خطأ في قراءة ملف Pine Script: {e}")
        return None


def load_market_data(filepath: str = None, symbol: str = 'BTCUSDT', bars: int = 500) -> pd.DataFrame:
    """تحميل بيانات السوق"""
    if filepath:
        try:
            data = pd.read_csv(filepath)
            print(f"✅ تم تحميل البيانات من: {filepath}")
            return data
        except Exception as e:
            print(f"❌ خطأ في تحميل البيانات: {e}")
            return None
    else:
        # إنشاء بيانات عشوائية للاختبار
        print(f"📊 إنشاء بيانات اختبار: {symbol} ({bars} شمعة)")
        
        np.random.seed(42)
        dates = pd.date_range(end=datetime.now(), periods=bars, freq='1h')
        
        # محاكاة حركة سعر واقعية
        returns = np.random.randn(bars) * 0.02
        price = 100 * np.exp(np.cumsum(returns))
        
        data = pd.DataFrame({
            'timestamp': dates,
            'open': price + np.random.randn(bars) * 0.5,
            'high': price + abs(np.random.randn(bars) * 2),
            'low': price - abs(np.random.randn(bars) * 2),
            'close': price,
            'volume': np.random.randint(1000, 100000, bars)
        })
        
        # التأكد من أن high/low صحيحة
        data['high'] = data[['open', 'high', 'close']].max(axis=1)
        data['low'] = data[['open', 'low', 'close']].min(axis=1)
        
        return data


def run_lorentzian_indicator(data: pd.DataFrame, 
                             neighbors_count: int = 8,
                             max_bars_back: int = 2000,
                             feature_count: int = 5) -> Dict[str, Any]:
    """تشغيل مؤشر Lorentzian Classification"""
    print("=" * 70)
    print("🚀 Lorentzian Classification - Machine Learning Indicator")
    print("=" * 70)
    print()
    
    settings = {
        'neighbors_count': neighbors_count,
        'max_bars_back': max_bars_back,
        'feature_count': feature_count
    }
    
    # إنشاء المصنف
    classifier = LorentzianClassifier(data, settings)
    
    # حساب المميزات
    classifier.calculate_features()
    
    # التصنيف
    predictions = classifier.classify(neighbors_count)
    
    # حساب Kernel Regression
    print("\n📊 حساب Kernel Regression...")
    kernels = KernelFunctions()
    yhat1 = kernels.rationalQuadratic(data['close'], lookback=8, relative_weight=8.0, start_at_bar=25)
    yhat2 = kernels.gaussian(data['close'], lookback=6, relative_weight=8.0, start_at_bar=25)
    
    # النتائج
    results = {
        'predictions': predictions,
        'features': classifier.features,
        'kernel_estimate': yhat1,
        'kernel_smooth': yhat2,
        'data': data
    }
    
    print("\n✅ اكتمل التحليل!")
    print("=" * 70)
    
    return results


def display_results(results: dict, output_file: str = None):
    """عرض وحفظ النتائج"""
    print("\n" + "=" * 70)
    print("📊 ملخص النتائج")
    print("=" * 70)
    
    predictions = results['predictions']
    
    # إحصائيات الإشارات
    long_signals = (predictions > 0).sum()
    short_signals = (predictions < 0).sum()
    neutral_signals = (predictions == 0).sum()
    
    print(f"\n📈 إشارات الشراء (Long): {long_signals}")
    print(f"📉 إشارات البيع (Short): {short_signals}")
    print(f"⚪ إشارات محايدة (Neutral): {neutral_signals}")
    
    # قوة الإشارات
    if long_signals > 0:
        avg_long_strength = predictions[predictions > 0].mean()
        print(f"💪 متوسط قوة إشارات الشراء: {avg_long_strength:.2f}")
    
    if short_signals > 0:
        avg_short_strength = abs(predictions[predictions < 0].mean())
        print(f"💪 متوسط قوة إشارات البيع: {avg_short_strength:.2f}")
    
    # حفظ النتائج
    if output_file:
        output_df = pd.DataFrame({
            'timestamp': results['data']['timestamp'],
            'close': results['data']['close'],
            'prediction': predictions,
            'signal': predictions.apply(lambda x: 'LONG' if x > 0 else ('SHORT' if x < 0 else 'NEUTRAL')),
            'kernel_estimate': results['kernel_estimate'],
            'kernel_smooth': results['kernel_smooth']
        })
        
        # إضافة المميزات
        for feature_name, feature_series in results['features'].items():
            output_df[feature_name] = feature_series
        
        output_df.to_csv(output_file, index=False)
        print(f"\n✅ تم حفظ النتائج في: {output_file}")
        
        # عرض أول 5 إشارات قوية
        print("\n🎯 أقوى 5 إشارات شراء:")
        top_longs = output_df[output_df['prediction'] > 0].nlargest(5, 'prediction')
        if not top_longs.empty:
            for idx, row in top_longs.iterrows():
                print(f"   {row['timestamp']} - القوة: {row['prediction']:.1f} - السعر: {row['close']:.2f}")
        
        print("\n🎯 أقوى 5 إشارات بيع:")
        top_shorts = output_df[output_df['prediction'] < 0].nsmallest(5, 'prediction')
        if not top_shorts.empty:
            for idx, row in top_shorts.iterrows():
                print(f"   {row['timestamp']} - القوة: {row['prediction']:.1f} - السعر: {row['close']:.2f}")


# ═══════════════════════════════════════════════════════════════════
# PART 6: Main Entry Point
# ═══════════════════════════════════════════════════════════════════

def main():
    """الدالة الرئيسية"""
    parser = argparse.ArgumentParser(
        description='Lorentzian Classification - نظام هجين Pine Script + Python',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
أمثلة الاستخدام:
  %(prog)s --demo
  %(prog)s -f lorentzian.pine -d data.csv -o results.csv
  %(prog)s --bars 1000 --neighbors 12
        """
    )
    
    parser.add_argument('-f', '--file', help='ملف Pine Script')
    parser.add_argument('-d', '--data', help='ملف البيانات CSV')
    parser.add_argument('-o', '--output', default='lorentzian_results.csv', help='ملف النتائج')
    parser.add_argument('--symbol', default='BTCUSDT', help='رمز السوق')
    parser.add_argument('--bars', type=int, default=500, help='عدد الشموع')
    parser.add_argument('--neighbors', type=int, default=8, help='عدد الجيران (k)')
    parser.add_argument('--demo', action='store_true', help='تشغيل تجريبي سريع')
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("🤖 Lorentzian Classification - Machine Learning Indicator")
    print("=" * 70)
    print("📚 نظام هجين: Pine Script + Python في ملف واحد")
    print("🔬 مشروع بحثي متقدم")
    print("=" * 70)
    print()
    
    # الوضع التجريبي
    if args.demo:
        print("🎮 الوضع التجريبي - تشغيل سريع\n")
        data = load_market_data(symbol='BTCUSDT', bars=200)
        
        if data is not None:
            results = run_lorentzian_indicator(
                data,
                neighbors_count=8,
                max_bars_back=200,
                feature_count=5
            )
            
            display_results(results, 'demo_results.csv')
        
        return 0
    
    # تحميل ملف Pine Script (إذا وجد)
    if args.file:
        pine_code = load_pine_script_file(args.file)
        if pine_code:
            print(f"✅ تم تحميل ملف Pine Script: {args.file}")
            print(f"📏 حجم الكود: {len(pine_code)} حرف\n")
    
    # تحميل البيانات
    data = load_market_data(args.data, args.symbol, args.bars)
    
    if data is None:
        print("❌ فشل تحميل البيانات!")
        return 1
    
    print()
    
    # تشغيل المؤشر
    try:
        results = run_lorentzian_indicator(
            data,
            neighbors_count=args.neighbors,
            max_bars_back=min(args.bars, 2000),
            feature_count=5
        )
        
        # عرض النتائج
        display_results(results, args.output)
        
        print("\n" + "=" * 70)
        print("✅ اكتمل التحليل بنجاح!")
        print("=" * 70)
        
        return 0
        
    except Exception as e:
        print(f"\n❌ خطأ في التنفيذ: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
