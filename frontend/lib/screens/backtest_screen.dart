import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import '../providers/newspaper_provider.dart';

class BacktestScreen extends StatefulWidget {
  const BacktestScreen({super.key});

  @override
  State<BacktestScreen> createState() => _BacktestScreenState();
}

class _BacktestScreenState extends State<BacktestScreen> {
  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addPostFrameCallback((_) {
      final provider = Provider.of<NewspaperProvider>(context, listen: false);
      if (provider.backtestSummary.isEmpty) {
        provider.loadBacktestSummary();
      }
    });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text(
          'Historical Performance',
          style: TextStyle(fontWeight: FontWeight.bold),
        ),
        actions: [
          Consumer<NewspaperProvider>(
            builder: (context, provider, child) {
              return IconButton(
                icon: provider.isLoading
                    ? const SizedBox(
                        width: 20,
                        height: 20,
                        child: CircularProgressIndicator(
                          strokeWidth: 2,
                          valueColor: AlwaysStoppedAnimation<Color>(Colors.white),
                        ),
                      )
                    : const Icon(Icons.refresh),
                onPressed: provider.isLoading ? null : () => provider.loadBacktestSummary(),
              );
            },
          ),
        ],
      ),
      body: Consumer<NewspaperProvider>(
        builder: (context, provider, child) {
          final backtestData = provider.backtestSummary;

          if (backtestData.isEmpty) {
            return const Center(
              child: Column(
                mainAxisAlignment: MainAxisAlignment.center,
                children: [
                  Icon(
                    Icons.analytics_outlined,
                    size: 64,
                    color: Colors.grey,
                  ),
                  SizedBox(height: 16),
                  Text(
                    'No backtest data available',
                    style: TextStyle(
                      fontSize: 16,
                      color: Colors.grey,
                    ),
                  ),
                  SizedBox(height: 8),
                  Text(
                    'Run a backtest from the API to see results',
                    style: TextStyle(
                      fontSize: 14,
                      color: Colors.grey,
                    ),
                  ),
                ],
              ),
            );
          }

          return RefreshIndicator(
            onRefresh: () => provider.loadBacktestSummary(),
            child: SingleChildScrollView(
              physics: const AlwaysScrollableScrollPhysics(),
              padding: const EdgeInsets.all(16),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  // Summary Stats
                  _buildSummaryStats(backtestData),
                  const SizedBox(height: 16),

                  // Top Performers
                  _buildTopPerformers(backtestData),
                  const SizedBox(height: 16),

                  // All Results
                  _buildAllResults(backtestData),
                  const SizedBox(height: 100), // Extra space for bottom nav
                ],
              ),
            ),
          );
        },
      ),
    );
  }

  Widget _buildSummaryStats(List backtestData) {
    final performers = backtestData.where((b) => b.isPerformer).length;
    final avgReturn = backtestData.isEmpty 
        ? 0.0 
        : backtestData.map((b) => b.returnPct).reduce((a, b) => a + b) / backtestData.length;
    
    return Card(
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            const Text(
              'Backtest Summary',
              style: TextStyle(
                fontSize: 18,
                fontWeight: FontWeight.bold,
              ),
            ),
            const SizedBox(height: 12),
            Row(
              mainAxisAlignment: MainAxisAlignment.spaceEvenly,
              children: [
                _buildStatItem(
                  'Total Tested',
                  backtestData.length.toString(),
                  Colors.blue,
                ),
                _buildStatItem(
                  'Top Performers',
                  performers.toString(),
                  Colors.green,
                ),
                _buildStatItem(
                  'Avg Return',
                  '${avgReturn.toStringAsFixed(1)}%',
                  avgReturn > 0 ? Colors.green : Colors.red,
                ),
              ],
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildTopPerformers(List backtestData) {
    final topPerformers = backtestData
        .where((b) => b.isPerformer)
        .take(5)
        .toList();

    if (topPerformers.isEmpty) {
      return const SizedBox.shrink();
    }

    return Card(
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            const Text(
              'Top Performers',
              style: TextStyle(
                fontSize: 18,
                fontWeight: FontWeight.bold,
              ),
            ),
            const SizedBox(height: 12),
            ...topPerformers.map((backtest) => _buildBacktestItem(backtest, true)),
          ],
        ),
      ),
    );
  }

  Widget _buildAllResults(List backtestData) {
    return Card(
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            const Text(
              'All Results',
              style: TextStyle(
                fontSize: 18,
                fontWeight: FontWeight.bold,
              ),
            ),
            const SizedBox(height: 12),
            ...backtestData.map((backtest) => _buildBacktestItem(backtest, false)),
          ],
        ),
      ),
    );
  }

  Widget _buildBacktestItem(backtest, bool isTopPerformer) {
    return Container(
      margin: const EdgeInsets.only(bottom: 8),
      padding: const EdgeInsets.all(12),
      decoration: BoxDecoration(
        color: isTopPerformer ? Colors.green[50] : Colors.grey[50],
        borderRadius: BorderRadius.circular(8),
        border: Border.all(
          color: isTopPerformer ? Colors.green[200]! : Colors.grey[300]!,
        ),
      ),
      child: Column(
        children: [
          Row(
            mainAxisAlignment: MainAxisAlignment.spaceBetween,
            children: [
              Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(
                    backtest.ticker,
                    style: const TextStyle(
                      fontSize: 16,
                      fontWeight: FontWeight.bold,
                    ),
                  ),
                  Text(
                    'Grade: ${backtest.performanceGrade}',
                    style: TextStyle(
                      fontSize: 12,
                      color: Colors.grey[600],
                    ),
                  ),
                ],
              ),
              Container(
                padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
                decoration: BoxDecoration(
                  color: _getReturnColor(backtest.returnPct),
                  borderRadius: BorderRadius.circular(12),
                ),
                child: Text(
                  '${backtest.returnPct.toStringAsFixed(1)}%',
                  style: const TextStyle(
                    fontSize: 12,
                    fontWeight: FontWeight.bold,
                    color: Colors.white,
                  ),
                ),
              ),
            ],
          ),
          const SizedBox(height: 8),
          Row(
            mainAxisAlignment: MainAxisAlignment.spaceEvenly,
            children: [
              _buildMetric('Sharpe', backtest.sharpeRatio.toStringAsFixed(2)),
              _buildMetric('Win Rate', '${backtest.winRate.toStringAsFixed(1)}%'),
              _buildMetric('Trades', backtest.totalTrades.toString()),
              _buildMetric('Premium', backtest.premiumCollectedFormatted),
            ],
          ),
        ],
      ),
    );
  }

  Widget _buildStatItem(String label, String value, Color color) {
    return Column(
      children: [
        Container(
          padding: const EdgeInsets.all(12),
          decoration: BoxDecoration(
            color: color.withOpacity(0.1),
            borderRadius: BorderRadius.circular(8),
          ),
          child: Text(
            value,
            style: TextStyle(
              fontSize: 20,
              fontWeight: FontWeight.bold,
              color: color,
            ),
          ),
        ),
        const SizedBox(height: 4),
        Text(
          label,
          style: const TextStyle(
            fontSize: 12,
            color: Colors.grey,
          ),
        ),
      ],
    );
  }

  Widget _buildMetric(String label, String value) {
    return Column(
      children: [
        Text(
          value,
          style: const TextStyle(
            fontSize: 12,
            fontWeight: FontWeight.w600,
          ),
        ),
        Text(
          label,
          style: const TextStyle(
            fontSize: 10,
            color: Colors.grey,
          ),
        ),
      ],
    );
  }

  Color _getReturnColor(double returnPct) {
    if (returnPct >= 20) return Colors.green;
    if (returnPct >= 10) return Colors.orange;
    if (returnPct >= 0) return Colors.yellow[700]!;
    return Colors.red;
  }
}