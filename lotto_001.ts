import axios from 'axios';
import * as fs from 'fs';
import { promisify } from 'util';
import AdmZip from 'adm-zip';
import * as ss from 'simple-statistics';
import { Matrix } from 'ml-matrix';
import { RandomForestRegression } from 'ml-random-forest';
import ConfusionMatrix from 'ml-confusion-matrix';

const writeFileAsync = promisify(fs.writeFile);

interface LottoResult {
  orderNumber: number;
  date: Date;
  numbers: number[];
}

// 1. Data Download and Processing
async function downloadLottoResults(): Promise<string> {
  try {
    const response = await axios({
      url: 'http://www.mbnet.com.pl/bazalosl.zip',
      method: 'GET',
      responseType: 'arraybuffer'
    });
    
    await writeFileAsync('lotto_results.zip', response.data);
    return 'lotto_results.zip';
  } catch (error: unknown) {
    if (error instanceof Error) {
      throw new Error(`Failed to download: ${error.message}`);
    }
    throw new Error('Failed to download: Unknown error');
  }
}

function extractAndParseResults(zipPath: string): LottoResult[] {
  if (!fs.existsSync(zipPath)) {
    throw new Error('Zip file not found!');
  }
  
  const zip = new AdmZip(zipPath);
  const entries = zip.getEntries();
  const dataEntry = entries.find(entry => !entry.isDirectory && entry.header.size > 0);
  
  if (!dataEntry) {
    throw new Error('No valid data file found in zip');
  }
  
  const content = dataEntry.getData().toString('utf8');
  const lines = content.split('\n').filter(line => line.trim());
  
  return lines
    .map(line => {
      try {
        const parts = line.trim().split(' ');
        if (parts.length < 3) return null;
        
        const orderNumber = parseInt(parts[0].replace('.', ''), 10);
        const dateParts = parts[1].split('.');
        const date = new Date(
          parseInt(dateParts[2], 10) + (parseInt(dateParts[2], 10) < 100 ? 2000 : 0),
          parseInt(dateParts[1], 10) - 1,
          parseInt(dateParts[0], 10)
        );
        const numbers = parts[2].split(',').map(n => parseInt(n, 10));
        
        return { orderNumber, date, numbers };
      } catch (error) {
        return null;
      }
    })
    .filter((result): result is LottoResult => result !== null)
    .sort((a, b) => b.date.getTime() - a.date.getTime());
}

// 2. Analysis Methods
class LottoAnalyzer {
  private results: LottoResult[];
  
  constructor(results: LottoResult[]) {
    this.results = results;
  }
  
  // Method 1: Frequency Analysis with Time Weighting
  analyzeFrequency(months: number = 12): Map<number, number> {
    const frequency = new Map<number, number>();
    const cutoffDate = new Date();
    cutoffDate.setMonth(cutoffDate.getMonth() - months);
    
    this.results
      .filter(r => r.date >= cutoffDate)
      .forEach(result => {
        result.numbers.forEach(num => {
          const weight = 1 - (new Date().getTime() - result.date.getTime()) / (months * 30 * 24 * 60 * 60 * 1000);
          frequency.set(num, (frequency.get(num) || 0) + weight);
        });
      });
    
    return frequency;
  }
  
  // Method 2: Gap Analysis with Due Numbers
  analyzeGaps(): Map<number, { lastAppearance: number; avgGap: number }> {
    const gaps = new Map<number, number[]>();
    const lastAppearances = new Map<number, number>();
    
    this.results.forEach((result, index) => {
      result.numbers.forEach(num => {
        if (lastAppearances.has(num)) {
          const gap = index - lastAppearances.get(num)!;
          gaps.set(num, [...(gaps.get(num) || []), gap]);
        }
        lastAppearances.set(num, index);
      });
    });
    
    const analysis = new Map<number, { lastAppearance: number; avgGap: number }>();
    gaps.forEach((gapsList, num) => {
      analysis.set(num, {
        lastAppearance: lastAppearances.get(num)!,
        avgGap: ss.mean(gapsList)
      });
    });
    
    return analysis;
  }
  
  // Method 3: Pattern Recognition with Position Analysis
  analyzePatterns(): Map<string, number> {
    const patterns = new Map<string, number>();
    
    this.results.forEach(result => {
      // Analyze number pairs
      for (let i = 0; i < result.numbers.length; i++) {
        for (let j = i + 1; j < result.numbers.length; j++) {
          const pair = `${result.numbers[i]},${result.numbers[j]}`;
          patterns.set(pair, (patterns.get(pair) || 0) + 1);
        }
      }
      
      // Analyze positional patterns
      result.numbers.forEach((num, pos) => {
        const posPattern = `pos${pos}:${num}`;
        patterns.set(posPattern, (patterns.get(posPattern) || 0) + 1);
      });
    });
    
    return patterns;
  }
  
  // Method 4: Statistical Distribution with Sum Analysis
  analyzeStatistics(): {
    numberDistribution: Map<number, number>;
    sumDistribution: Map<number, number>;
    rangeDistribution: Map<string, number>;
  } {
    const numberDistribution = new Map<number, number>();
    const sumDistribution = new Map<number, number>();
    const rangeDistribution = new Map<string, number>();
    
    this.results.forEach(result => {
      // Number distribution
      result.numbers.forEach(num => {
        numberDistribution.set(num, (numberDistribution.get(num) || 0) + 1);
      });
      
      // Sum distribution
      const sum = result.numbers.reduce((a, b) => a + b, 0);
      sumDistribution.set(sum, (sumDistribution.get(sum) || 0) + 1);
      
      // Range distribution
      const ranges = result.numbers.map(n => Math.floor((n - 1) / 10));
      const rangeKey = ranges.join(',');
      rangeDistribution.set(rangeKey, (rangeDistribution.get(rangeKey) || 0) + 1);
    });
    
    return { numberDistribution, sumDistribution, rangeDistribution };
  }
  
  // Method 5: Machine Learning with Random Forest
  async trainMLModel(): Promise<RandomForestRegression> {
    const features: number[][] = [];
    const labels: number[] = [];
    
    // Prepare training data
    for (let i = 0; i < this.results.length - 1; i++) {
      const currentDraw = this.results[i];
      const nextDraw = this.results[i + 1];
      
      // Features: current draw numbers, their positions, and statistics
      const feature = [
        ...currentDraw.numbers,
        ...currentDraw.numbers.map(n => n % 10), // Last digit
        ss.mean(currentDraw.numbers),
        ss.standardDeviation(currentDraw.numbers)
      ];
      
      features.push(feature);
      labels.push(nextDraw.numbers[0]); // Predict first number of next draw
    }
    
    // Train Random Forest
    const rf = new RandomForestRegression({
      nEstimators: 100,
      seed: 42
    });
    
    rf.train(features, labels);
    return rf;
  }
  
  // Generate predictions using all methods
  async generatePredictions(): Promise<{
    frequency: number[];
    gaps: number[];
    patterns: number[];
    statistical: number[];
    ml: number[];
  }> {
    // Frequency Analysis
    const frequency = this.analyzeFrequency();
    const frequencyPrediction = Array.from(frequency.entries())
      .sort((a, b) => b[1] - a[1])
      .slice(0, 6)
      .map(([num]) => num);
    
    // Gap Analysis
    const gaps = this.analyzeGaps();
    const gapsPrediction = Array.from(gaps.entries())
      .sort((a, b) => b[1].lastAppearance - a[1].lastAppearance)
      .slice(0, 6)
      .map(([num]) => num);
    
    // Pattern Analysis
    const patterns = this.analyzePatterns();
    const patternPrediction = Array.from(patterns.entries())
      .sort((a, b) => b[1] - a[1])
      .slice(0, 6)
      .map(([pattern]) => parseInt(pattern.split(',')[0]));
    
    // Statistical Analysis
    const stats = this.analyzeStatistics();
    const statisticalPrediction = Array.from(stats.numberDistribution.entries())
      .sort((a, b) => b[1] - a[1])
      .slice(0, 6)
      .map(([num]) => num);
    
    // Machine Learning
    const model = await this.trainMLModel();
    const lastDraw = this.results[0];
    const mlFeatures = [
      ...lastDraw.numbers,
      ...lastDraw.numbers.map(n => n % 10),
      ss.mean(lastDraw.numbers),
      ss.standardDeviation(lastDraw.numbers)
    ];
    const mlPrediction = Array.from({ length: 6 }, () => 
      Math.round(model.predict([mlFeatures])[0])
    );
    
    return {
      frequency: frequencyPrediction,
      gaps: gapsPrediction,
      patterns: patternPrediction,
      statistical: statisticalPrediction,
      ml: mlPrediction
    };
  }
}

// Main execution
async function main() {
  try {
    console.log('Starting lottery analysis...\n');
    
    // Download and process data
    console.log('1. Downloading lottery results...');
    const zipPath = await downloadLottoResults();
    console.log('✓ Downloaded zip file:', zipPath);
    
    console.log('\n2. Parsing historical data...');
    const results = extractAndParseResults(zipPath);
    console.log('✓ Total results parsed:', results.length);
    
    // Initialize analyzer
    console.log('\n3. Initializing analysis engine...');
    const analyzer = new LottoAnalyzer(results);
    console.log('✓ Analysis engine ready');
    
    // Generate predictions with progress
    console.log('\n4. Generating predictions...');
    console.log('   - Running frequency analysis...');
    const frequency = analyzer.analyzeFrequency();
    console.log('   - Analyzing number gaps...');
    const gaps = analyzer.analyzeGaps();
    console.log('   - Identifying patterns...');
    const patterns = analyzer.analyzePatterns();
    console.log('   - Calculating statistics...');
    const stats = analyzer.analyzeStatistics();
    console.log('   - Training machine learning model...');
    const model = await analyzer.trainMLModel();
    console.log('✓ All analysis methods completed');
    
    // Generate final predictions
    console.log('\n5. Compiling final predictions...');
    const predictions = await analyzer.generatePredictions();
    
    // Display detailed analysis
    console.log('\n=== DETAILED LOTTERY ANALYSIS ===');
    
    // Last draw information
    const lastDraw = results[0];
    console.log('\nLast Draw:', lastDraw.date.toLocaleDateString());
    console.log('Numbers:', lastDraw.numbers.join(', '));
    
    // Frequency Analysis
    console.log('\n--- FREQUENCY ANALYSIS (Last 12 months) ---');
    const topFrequent = Array.from(frequency.entries())
      .sort((a, b) => b[1] - a[1])
      .slice(0, 10);
    console.log('Most frequent numbers:');
    topFrequent.forEach(([num, freq]) => {
      console.log(`Number ${num}: ${freq.toFixed(2)} weighted appearances`);
    });
    
    // Gap Analysis
    console.log('\n--- GAP ANALYSIS ---');
    const dueNumbers = Array.from(gaps.entries())
      .sort((a, b) => b[1].lastAppearance - a[1].lastAppearance)
      .slice(0, 10);
    console.log('Numbers due to appear (longest time since last appearance):');
    dueNumbers.forEach(([num, data]) => {
      const daysSinceLast = Math.floor((new Date().getTime() - results[data.lastAppearance].date.getTime()) / (1000 * 60 * 60 * 24));
      console.log(`Number ${num}: ${daysSinceLast} days since last appearance (avg gap: ${data.avgGap.toFixed(1)} draws)`);
    });
    
    // Pattern Analysis
    console.log('\n--- PATTERN ANALYSIS ---');
    const topPatterns = Array.from(patterns.entries())
      .filter(([pattern]) => pattern.startsWith('pos'))
      .sort((a, b) => b[1] - a[1])
      .slice(0, 6);
    console.log('Most common positional patterns:');
    topPatterns.forEach(([pattern, count]) => {
      const [pos, num] = pattern.split(':');
      console.log(`Position ${pos}: Number ${num} appeared ${count} times`);
    });
    
    // Statistical Analysis
    console.log('\n--- STATISTICAL ANALYSIS ---');
    const avgSum = Array.from(stats.sumDistribution.entries())
      .reduce((acc, [sum, count]) => acc + (sum * count), 0) / results.length;
    console.log(`Average sum of winning numbers: ${avgSum.toFixed(1)}`);
    
    // Predictions
    console.log('\n=== PREDICTIONS FOR NEXT DRAW ===');
    console.log('1. Frequency-based:', predictions.frequency.join(', '));
    console.log('2. Gap-based (due numbers):', predictions.gaps.join(', '));
    console.log('3. Pattern-based:', predictions.patterns.join(', '));
    console.log('4. Statistical:', predictions.statistical.join(', '));
    console.log('5. Machine Learning:', predictions.ml.join(', '));
    
    // Combined prediction
    console.log('\n6. Generating combined prediction...');
    const allPredictions = [
      ...predictions.frequency,
      ...predictions.gaps,
      ...predictions.patterns,
      ...predictions.statistical,
      ...predictions.ml
    ];
    
    const numberCounts = new Map<number, number>();
    allPredictions.forEach(num => {
      numberCounts.set(num, (numberCounts.get(num) || 0) + 1);
    });
    
    const combinedPrediction = Array.from(numberCounts.entries())
      .sort((a, b) => b[1] - a[1])
      .slice(0, 6)
      .map(([num]) => num)
      .sort((a, b) => a - b);
    
    console.log('\nCombined Prediction (numbers appearing in multiple methods):');
    console.log(combinedPrediction.join(', '));
    
    console.log('\n✓ Analysis complete!');
    
  } catch (error) {
    console.error('Error:', error);
  }
}

// Run the program
main().catch(console.error);
