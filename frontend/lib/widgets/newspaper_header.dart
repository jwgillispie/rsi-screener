import 'package:flutter/material.dart';
import 'package:intl/intl.dart';
import '../models/daily_newspaper.dart';

class NewspaperHeader extends StatelessWidget {
  final DailyNewspaper newspaper;

  const NewspaperHeader({
    super.key,
    required this.newspaper,
  });

  @override
  Widget build(BuildContext context) {
    final date = DateTime.tryParse(newspaper.date) ?? DateTime.now();
    
    return Container(
      width: double.infinity,
      padding: const EdgeInsets.all(20),
      decoration: BoxDecoration(
        color: const Color(0xFF1E3A8A),
        borderRadius: BorderRadius.circular(12),
        boxShadow: [
          BoxShadow(
            color: Colors.black.withOpacity(0.1),
            blurRadius: 8,
            offset: const Offset(0, 4),
          ),
        ],
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          // Header Title
          Row(
            mainAxisAlignment: MainAxisAlignment.spaceBetween,
            children: [
              const Expanded(
                child: Text(
                  'THE IRON CONDOR TIMES',
                  style: TextStyle(
                    color: Colors.white,
                    fontSize: 24,
                    fontWeight: FontWeight.bold,
                    letterSpacing: 1.2,
                  ),
                ),
              ),
              Container(
                padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 6),
                decoration: BoxDecoration(
                  color: Colors.white.withOpacity(0.2),
                  borderRadius: BorderRadius.circular(20),
                ),
                child: Text(
                  DateFormat('MMM dd, yyyy').format(date),
                  style: const TextStyle(
                    color: Colors.white,
                    fontSize: 12,
                    fontWeight: FontWeight.w600,
                  ),
                ),
              ),
            ],
          ),
          const SizedBox(height: 4),
          
          // Subtitle
          const Text(
            'Daily Options Trading Intelligence',
            style: TextStyle(
              color: Colors.white70,
              fontSize: 14,
              fontStyle: FontStyle.italic,
            ),
          ),
          const SizedBox(height: 16),
          
          // Main Headline
          Text(
            _generateHeadline(),
            style: const TextStyle(
              color: Colors.white,
              fontSize: 18,
              fontWeight: FontWeight.w600,
              height: 1.3,
            ),
          ),
          const SizedBox(height: 8),
          
          // Trading Recommendation
          Container(
            padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 6),
            decoration: BoxDecoration(
              color: _getRecommendationColor(),
              borderRadius: BorderRadius.circular(6),
            ),
            child: Text(
              newspaper.tradingRecommendation.toUpperCase(),
              style: const TextStyle(
                color: Colors.white,
                fontSize: 12,
                fontWeight: FontWeight.bold,
                letterSpacing: 0.5,
              ),
            ),
          ),
        ],
      ),
    );
  }

  String _generateHeadline() {
    if (newspaper.totalOpportunities == 0) {
      return 'Markets Quiet: No Iron Condor Opportunities Today';
    } else if (newspaper.highConfidenceCount >= 3) {
      return 'Strong Trading Day: ${newspaper.highConfidenceCount} High-Confidence Opportunities Available';
    } else if (newspaper.totalOpportunities >= 5) {
      return 'Moderate Activity: ${newspaper.totalOpportunities} Iron Condor Opportunities Identified';
    } else {
      return 'Limited Opportunities: ${newspaper.totalOpportunities} Signals for Selective Trading';
    }
  }

  Color _getRecommendationColor() {
    if (newspaper.highConfidenceCount >= 3) {
      return Colors.green[600]!;
    } else if (newspaper.highConfidenceCount >= 1) {
      return Colors.orange[600]!;
    } else if (newspaper.totalOpportunities >= 3) {
      return Colors.yellow[700]!;
    } else {
      return Colors.red[600]!;
    }
  }
}