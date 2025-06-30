import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import 'package:intl/intl.dart';
import '../providers/newspaper_provider.dart';
import '../widgets/newspaper_header.dart';
import '../widgets/market_summary_card.dart';
import '../widgets/top_signals_card.dart';
import '../widgets/loading_widget.dart';
import '../widgets/error_widget.dart' as custom_widgets;

class HomeScreen extends StatefulWidget {
  const HomeScreen({super.key});

  @override
  State<HomeScreen> createState() => _HomeScreenState();
}

class _HomeScreenState extends State<HomeScreen> {
  @override
  void initState() {
    super.initState();
    // Load data when screen initializes
    WidgetsBinding.instance.addPostFrameCallback((_) {
      _loadData();
    });
  }

  void _loadData() {
    final provider = Provider.of<NewspaperProvider>(context, listen: false);
    if (!provider.hasData || provider.needsRefresh) {
      provider.refresh();
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text(
          'Iron Condor Trading Newspaper',
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
                onPressed: provider.isLoading ? null : () => provider.refresh(),
              );
            },
          ),
        ],
      ),
      body: Consumer<NewspaperProvider>(
        builder: (context, provider, child) {
          if (provider.isLoading && !provider.hasData) {
            return const LoadingWidget(message: 'Loading daily newspaper...');
          }

          if (provider.hasError && !provider.hasData) {
            return custom_widgets.ErrorWidget(
              message: provider.error!,
              onRetry: () => provider.refresh(),
            );
          }

          if (provider.newspaper == null) {
            return const Center(
              child: Text(
                'No data available',
                style: TextStyle(fontSize: 16, color: Colors.grey),
              ),
            );
          }

          return RefreshIndicator(
            onRefresh: () => provider.refresh(),
            child: SingleChildScrollView(
              physics: const AlwaysScrollableScrollPhysics(),
              padding: const EdgeInsets.all(16),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  // Newspaper Header
                  NewspaperHeader(newspaper: provider.newspaper!),
                  const SizedBox(height: 16),

                  // Market Summary
                  MarketSummaryCard(newspaper: provider.newspaper!),
                  const SizedBox(height: 16),

                  // Top Signals
                  TopSignalsCard(signals: provider.newspaper!.topSignals),
                  const SizedBox(height: 16),

                  // Market Conditions
                  _buildMarketConditionsCard(provider.newspaper!),
                  const SizedBox(height: 16),

                  // Last Updated
                  _buildLastUpdatedCard(provider.lastRefresh),
                  const SizedBox(height: 100), // Extra space for bottom nav
                ],
              ),
            ),
          );
        },
      ),
    );
  }

  Widget _buildMarketConditionsCard(newspaper) {
    return Card(
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              children: [
                const Icon(Icons.cloud, color: Color(0xFF1E3A8A)),
                const SizedBox(width: 8),
                const Text(
                  'Market Conditions',
                  style: TextStyle(
                    fontSize: 18,
                    fontWeight: FontWeight.bold,
                  ),
                ),
              ],
            ),
            const SizedBox(height: 12),
            _buildConditionRow(
              'Volatility Regime',
              newspaper.marketConditions.volatilityDescription,
              _getVolatilityColor(newspaper.marketConditions.volatilityRegime),
            ),
            const SizedBox(height: 8),
            _buildConditionRow(
              'Market Bias',
              newspaper.marketConditions.marketBias.toUpperCase(),
              Colors.blue,
            ),
            const SizedBox(height: 8),
            _buildConditionRow(
              'Total Premium Available',
              '\$${NumberFormat('#,##0.00').format(newspaper.marketConditions.totalPremiumAvailable)}',
              Colors.green,
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildConditionRow(String label, String value, Color valueColor) {
    return Row(
      mainAxisAlignment: MainAxisAlignment.spaceBetween,
      children: [
        Text(
          label,
          style: const TextStyle(fontSize: 14, color: Colors.grey),
        ),
        Text(
          value,
          style: TextStyle(
            fontSize: 14,
            fontWeight: FontWeight.w600,
            color: valueColor,
          ),
        ),
      ],
    );
  }

  Color _getVolatilityColor(String regime) {
    switch (regime.toLowerCase()) {
      case 'very_high':
        return Colors.red;
      case 'high':
        return Colors.orange;
      case 'moderate':
        return Colors.yellow[700]!;
      case 'low':
        return Colors.grey;
      default:
        return Colors.grey;
    }
  }

  Widget _buildLastUpdatedCard(DateTime? lastRefresh) {
    return Card(
      color: Colors.grey[100],
      child: Padding(
        padding: const EdgeInsets.all(12),
        child: Row(
          children: [
            const Icon(Icons.access_time, size: 16, color: Colors.grey),
            const SizedBox(width: 8),
            Text(
              lastRefresh != null
                  ? 'Last updated: ${DateFormat('MMM dd, yyyy \'at\' HH:mm').format(lastRefresh)}'
                  : 'Not updated',
              style: const TextStyle(fontSize: 12, color: Colors.grey),
            ),
          ],
        ),
      ),
    );
  }
}