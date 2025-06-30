import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import '../providers/newspaper_provider.dart';
import '../widgets/top_signals_card.dart';
import '../widgets/loading_widget.dart';
import '../widgets/error_widget.dart' as custom_widgets;

class SignalsScreen extends StatelessWidget {
  const SignalsScreen({super.key});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text(
          'Iron Condor Signals',
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
            return const LoadingWidget(message: 'Loading signals...');
          }

          if (provider.hasError && !provider.hasData) {
            return custom_widgets.ErrorWidget(
              message: provider.error!,
              onRetry: () => provider.refresh(),
            );
          }

          final signals = provider.newspaper?.signals ?? [];

          return RefreshIndicator(
            onRefresh: () => provider.refresh(),
            child: SingleChildScrollView(
              physics: const AlwaysScrollableScrollPhysics(),
              padding: const EdgeInsets.all(16),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  // Header Stats
                  _buildHeaderStats(signals),
                  const SizedBox(height: 16),

                  // All Signals
                  TopSignalsCard(signals: signals),
                  const SizedBox(height: 100), // Extra space for bottom nav
                ],
              ),
            ),
          );
        },
      ),
    );
  }

  Widget _buildHeaderStats(List signals) {
    final highConfidence = signals.where((s) => s.confidenceScore >= 80).length;
    final goodConfidence = signals.where((s) => s.confidenceScore >= 60 && s.confidenceScore < 80).length;
    
    return Card(
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            const Text(
              'Signal Overview',
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
                  'Total',
                  signals.length.toString(),
                  Colors.blue,
                ),
                _buildStatItem(
                  'High Confidence',
                  highConfidence.toString(),
                  Colors.green,
                ),
                _buildStatItem(
                  'Good Confidence',
                  goodConfidence.toString(),
                  Colors.orange,
                ),
              ],
            ),
          ],
        ),
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
              fontSize: 24,
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
}