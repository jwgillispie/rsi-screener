import 'package:json_annotation/json_annotation.dart';
import 'iron_condor_signal.dart';

part 'daily_newspaper.g.dart';

@JsonSerializable()
class MarketConditions {
  @JsonKey(name: 'volatility_regime')
  final String volatilityRegime;
  @JsonKey(name: 'total_premium_available')
  final double totalPremiumAvailable;
  @JsonKey(name: 'market_bias')
  final String marketBias;

  MarketConditions({
    required this.volatilityRegime,
    required this.totalPremiumAvailable,
    required this.marketBias,
  });

  factory MarketConditions.fromJson(Map<String, dynamic> json) =>
      _$MarketConditionsFromJson(json);

  Map<String, dynamic> toJson() => _$MarketConditionsToJson(this);

  String get volatilityDescription {
    switch (volatilityRegime.toLowerCase()) {
      case 'very_high':
        return 'Very High - Excellent for Iron Condors';
      case 'high':
        return 'High - Good Opportunity';
      case 'moderate':
        return 'Moderate - Fair Conditions';
      case 'low':
        return 'Low - Poor Conditions';
      default:
        return 'Unknown';
    }
  }
}

@JsonSerializable()
class DailyNewspaper {
  final String date;
  @JsonKey(name: 'market_summary')
  final String marketSummary;
  @JsonKey(name: 'total_opportunities')
  final int totalOpportunities;
  @JsonKey(name: 'high_confidence_count')
  final int highConfidenceCount;
  @JsonKey(name: 'avg_iv_rank')
  final double avgIvRank;
  final List<IronCondorSignal> signals;
  @JsonKey(name: 'market_conditions')
  final MarketConditions marketConditions;

  DailyNewspaper({
    required this.date,
    required this.marketSummary,
    required this.totalOpportunities,
    required this.highConfidenceCount,
    required this.avgIvRank,
    required this.signals,
    required this.marketConditions,
  });

  factory DailyNewspaper.fromJson(Map<String, dynamic> json) =>
      _$DailyNewspaperFromJson(json);

  Map<String, dynamic> toJson() => _$DailyNewspaperToJson(this);

  // Helper methods for UI
  List<IronCondorSignal> get highConfidenceSignals =>
      signals.where((signal) => signal.confidenceScore >= 80).toList();

  List<IronCondorSignal> get topSignals =>
      signals.take(5).toList();

  String get marketSentiment {
    if (totalOpportunities >= 10) return 'Bullish for Iron Condors';
    if (totalOpportunities >= 5) return 'Moderate Opportunities';
    if (totalOpportunities >= 1) return 'Limited Opportunities';
    return 'No Opportunities';
  }

  String get tradingRecommendation {
    if (highConfidenceCount >= 3) {
      return 'Strong trading day - multiple high-confidence opportunities';
    } else if (highConfidenceCount >= 1) {
      return 'Selective trading - focus on high-confidence signals';
    } else if (totalOpportunities >= 3) {
      return 'Cautious trading - analyze signals carefully';
    } else {
      return 'Wait for better opportunities';
    }
  }
}