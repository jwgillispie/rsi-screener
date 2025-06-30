import 'package:flutter/material.dart';
import '../models/daily_newspaper.dart';

class MarketSummaryCard extends StatelessWidget {
  final DailyNewspaper newspaper;

  const MarketSummaryCard({
    super.key,
    required this.newspaper,
  });

  @override
  Widget build(BuildContext context) {
    return Card(
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            // Header
            Row(
              children: [
                const Icon(Icons.summarize, color: Color(0xFF1E3A8A)),
                const SizedBox(width: 8),
                const Text(
                  'Market Summary',
                  style: TextStyle(
                    fontSize: 18,
                    fontWeight: FontWeight.bold,
                  ),
                ),
              ],
            ),
            const SizedBox(height: 16),

            // Summary Text
            Text(
              newspaper.marketSummary,
              style: const TextStyle(
                fontSize: 15,
                height: 1.4,
                color: Colors.black87,
              ),
            ),
            const SizedBox(height: 16),

            // Key Metrics Grid
            Row(
              children: [
                Expanded(
                  child: _buildMetricItem(
                    'Total Opportunities',
                    newspaper.totalOpportunities.toString(),
                    Icons.trending_up,
                    _getOpportunityColor(newspaper.totalOpportunities),
                  ),
                ),
                Expanded(
                  child: _buildMetricItem(
                    'High Confidence',
                    newspaper.highConfidenceCount.toString(),
                    Icons.star,
                    _getConfidenceColor(newspaper.highConfidenceCount),
                  ),
                ),
              ],
            ),
            const SizedBox(height: 12),
            Row(
              children: [
                Expanded(
                  child: _buildMetricItem(
                    'Avg IV Rank',
                    '${newspaper.avgIvRank.toStringAsFixed(1)}%',
                    Icons.speed,
                    _getIvRankColor(newspaper.avgIvRank),
                  ),
                ),
                Expanded(
                  child: _buildMetricItem(
                    'Market Sentiment',
                    newspaper.marketSentiment.split(' ')[0], // First word
                    Icons.psychology,
                    _getSentimentColor(newspaper.marketSentiment),
                  ),
                ),
              ],
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildMetricItem(String label, String value, IconData icon, Color color) {
    return Container(
      padding: const EdgeInsets.all(12),
      margin: const EdgeInsets.all(4),
      decoration: BoxDecoration(
        color: color.withOpacity(0.1),
        borderRadius: BorderRadius.circular(8),
        border: Border.all(color: color.withOpacity(0.3)),
      ),
      child: Column(
        children: [
          Icon(icon, color: color, size: 20),
          const SizedBox(height: 4),
          Text(
            value,
            style: TextStyle(
              fontSize: 16,
              fontWeight: FontWeight.bold,
              color: color,
            ),
          ),
          const SizedBox(height: 2),
          Text(
            label,
            style: const TextStyle(
              fontSize: 10,
              color: Colors.grey,
            ),
            textAlign: TextAlign.center,
          ),
        ],
      ),
    );
  }

  Color _getOpportunityColor(int opportunities) {
    if (opportunities >= 10) return Colors.green;
    if (opportunities >= 5) return Colors.orange;
    if (opportunities >= 1) return Colors.yellow[700]!;
    return Colors.grey;
  }

  Color _getConfidenceColor(int highConfidence) {
    if (highConfidence >= 3) return Colors.green;
    if (highConfidence >= 1) return Colors.orange;
    return Colors.grey;
  }

  Color _getIvRankColor(double ivRank) {
    if (ivRank >= 80) return Colors.green;
    if (ivRank >= 70) return Colors.orange;
    if (ivRank >= 50) return Colors.yellow[700]!;
    return Colors.grey;
  }

  Color _getSentimentColor(String sentiment) {
    if (sentiment.toLowerCase().contains('bullish')) return Colors.green;
    if (sentiment.toLowerCase().contains('moderate')) return Colors.orange;
    if (sentiment.toLowerCase().contains('limited')) return Colors.yellow[700]!;
    return Colors.grey;
  }
}