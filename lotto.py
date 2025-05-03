import requests
import zipfile
import io
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import confusion_matrix
import statistics
from typing import List, Dict, Tuple
from dataclasses import dataclass
from io import BytesIO

@dataclass
class LottoResult:
    order_number: int
    date: datetime
    numbers: List[int]

class LottoAnalyzer:
    def __init__(self, results: List[LottoResult]):
        self.results = results

    def analyze_frequency(self, months: int = 12) -> Dict[int, float]:
        frequency = {}
        cutoff_date = datetime.now() - timedelta(days=months * 30)
        
        for result in [r for r in self.results if r.date >= cutoff_date]:
            weight = 1 - (datetime.now() - result.date).total_seconds() / (months * 30 * 24 * 60 * 60)
            for num in result.numbers:
                frequency[num] = frequency.get(num, 0) + weight
        
        return frequency

    def analyze_gaps(self) -> Dict[int, Dict[str, float]]:
        gaps = {}
        last_appearances = {}
        
        for idx, result in enumerate(self.results):
            for num in result.numbers:
                if num in last_appearances:
                    gap = idx - last_appearances[num]
                    gaps.setdefault(num, []).append(gap)
                last_appearances[num] = idx
        
        analysis = {}
        for num, gap_list in gaps.items():
            analysis[num] = {
                'last_appearance': last_appearances[num],
                'avg_gap': statistics.mean(gap_list)
            }
        
        return analysis

    def analyze_patterns(self) -> Dict[str, int]:
        patterns = {}
        
        for result in self.results:
            # Analyze number pairs
            for i in range(len(result.numbers)):
                for j in range(i + 1, len(result.numbers)):
                    pair = f"{result.numbers[i]},{result.numbers[j]}"
                    patterns[pair] = patterns.get(pair, 0) + 1
            
            # Analyze positional patterns
            for pos, num in enumerate(result.numbers):
                pos_pattern = f"pos{pos}:{num}"
                patterns[pos_pattern] = patterns.get(pos_pattern, 0) + 1
        
        return patterns

    def analyze_statistics(self) -> Dict[str, Dict]:
        number_distribution = {}
        sum_distribution = {}
        range_distribution = {}
        
        for result in self.results:
            # Number distribution
            for num in result.numbers:
                number_distribution[num] = number_distribution.get(num, 0) + 1
            
            # Sum distribution
            total = sum(result.numbers)
            sum_distribution[total] = sum_distribution.get(total, 0) + 1
            
            # Range distribution
            ranges = [str((n - 1) // 10) for n in result.numbers]
            range_key = ','.join(ranges)
            range_distribution[range_key] = range_distribution.get(range_key, 0) + 1
        
        return {
            'number_distribution': number_distribution,
            'sum_distribution': sum_distribution,
            'range_distribution': range_distribution
        }

    def train_ml_model(self) -> RandomForestRegressor:
        features = []
        labels = []
        
        for i in range(len(self.results) - 1):
            current_draw = self.results[i]
            next_draw = self.results[i + 1]
            
            feature = (
                current_draw.numbers + 
                [n % 10 for n in current_draw.numbers] +
                [statistics.mean(current_draw.numbers)] +
                [statistics.stdev(current_draw.numbers)]
            )
            
            features.append(feature)
            labels.append(next_draw.numbers[0])
        
        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        rf.fit(features, labels)
        return rf

    async def generate_predictions(self) -> Dict[str, List[int]]:
        # Frequency Analysis
        frequency = self.analyze_frequency()
        frequency_prediction = sorted(frequency.items(), key=lambda x: x[1], reverse=True)[:6]
        frequency_prediction = [num for num, _ in frequency_prediction]
        
        # Gap Analysis
        gaps = self.analyze_gaps()
        gaps_prediction = sorted(gaps.items(), key=lambda x: x[1]['last_appearance'], reverse=True)[:6]
        gaps_prediction = [num for num, _ in gaps_prediction]
        
        # Pattern Analysis
        patterns = self.analyze_patterns()
        pattern_prediction = sorted(patterns.items(), key=lambda x: x[1], reverse=True)[:6]
        pattern_prediction = [int(pattern.split(',')[0]) for pattern, _ in pattern_prediction]
        
        # Statistical Analysis
        stats = self.analyze_statistics()
        statistical_prediction = sorted(stats['number_distribution'].items(), key=lambda x: x[1], reverse=True)[:6]
        statistical_prediction = [num for num, _ in statistical_prediction]
        
        # Machine Learning
        model = self.train_ml_model()
        last_draw = self.results[0]
        ml_features = (
            last_draw.numbers + 
            [n % 10 for n in last_draw.numbers] +
            [statistics.mean(last_draw.numbers)] +
            [statistics.stdev(last_draw.numbers)]
        )
        ml_prediction = [round(model.predict([ml_features])[0]) for _ in range(6)]
        
        return {
            'frequency': frequency_prediction,
            'gaps': gaps_prediction,
            'patterns': pattern_prediction,
            'statistical': statistical_prediction,
            'ml': ml_prediction
        }

def download_lotto_results() -> bytes:
    url = 'http://www.mbnet.com.pl/bazalosl.zip'
    response = requests.get(url)
    if response.status_code != 200:
        raise Exception('Failed to download lottery results')
    return response.content

def extract_and_parse_results(zip_content: bytes) -> List[LottoResult]:
    results = []
    with zipfile.ZipFile(BytesIO(zip_content)) as zf:
        for file in zf.namelist():
            with zf.open(file) as f:
                content = f.read().decode('utf-8')
                for line in content.split('\n'):
                    if not line.strip():
                        continue
                    
                    try:
                        parts = line.strip().split()
                        if len(parts) < 3:
                            continue
                        
                        order_number = int(parts[0].replace('.', ''))
                        date_parts = parts[1].split('.')
                        date = datetime(
                            year=int(date_parts[2]) + (2000 if int(date_parts[2]) < 100 else 0),
                            month=int(date_parts[1]),
                            day=int(date_parts[0])
                        )
                        numbers = [int(n) for n in parts[2].split(',')]
                        
                        results.append(LottoResult(order_number, date, numbers))
                    except Exception as e:
                        continue
    
    return sorted(results, key=lambda x: x.date, reverse=True)

async def main():
    try:
        print('Starting lottery analysis...\n')
        
        print('1. Downloading lottery results...')
        zip_content = download_lotto_results()
        print('✓ Downloaded zip file')
        
        print('\n2. Parsing historical data...')
        results = extract_and_parse_results(zip_content)
        print(f'✓ Total results parsed: {len(results)}')
        
        analyzer = LottoAnalyzer(results)
        predictions = await analyzer.generate_predictions()
        
        print('\n=== PREDICTIONS FOR NEXT DRAW ===')
        for method, numbers in predictions.items():
            print(f'{method.capitalize()}-based: {", ".join(map(str, numbers))}')
        
        # Combined prediction
        all_predictions = []
        for pred_list in predictions.values():
            all_predictions.extend(pred_list)
        
        number_counts = {}
        for num in all_predictions:
            number_counts[num] = number_counts.get(num, 0) + 1
        
        combined_prediction = sorted(
            sorted(number_counts.items(), key=lambda x: x[1], reverse=True)[:6]
        )
        
        print('\nCombined Prediction:')
        print(', '.join(map(str, [num for num, _ in combined_prediction])))
        
    except Exception as e:
        print(f'Error: {str(e)}')

if __name__ == '__main__':
    import asyncio
    asyncio.run(main())
