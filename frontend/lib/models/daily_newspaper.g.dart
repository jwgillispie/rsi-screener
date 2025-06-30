// GENERATED CODE - DO NOT MODIFY BY HAND

part of 'daily_newspaper.dart';

// **************************************************************************
// JsonSerializableGenerator
// **************************************************************************

MarketConditions _$MarketConditionsFromJson(Map<String, dynamic> json) =>
    MarketConditions(
      volatilityRegime: json['volatility_regime'] as String,
      totalPremiumAvailable: (json['total_premium_available'] as num)
          .toDouble(),
      marketBias: json['market_bias'] as String,
    );

Map<String, dynamic> _$MarketConditionsToJson(MarketConditions instance) =>
    <String, dynamic>{
      'volatility_regime': instance.volatilityRegime,
      'total_premium_available': instance.totalPremiumAvailable,
      'market_bias': instance.marketBias,
    };

DailyNewspaper _$DailyNewspaperFromJson(Map<String, dynamic> json) =>
    DailyNewspaper(
      date: json['date'] as String,
      marketSummary: json['market_summary'] as String,
      totalOpportunities: (json['total_opportunities'] as num).toInt(),
      highConfidenceCount: (json['high_confidence_count'] as num).toInt(),
      avgIvRank: (json['avg_iv_rank'] as num).toDouble(),
      signals: (json['signals'] as List<dynamic>)
          .map((e) => IronCondorSignal.fromJson(e as Map<String, dynamic>))
          .toList(),
      marketConditions: MarketConditions.fromJson(
        json['market_conditions'] as Map<String, dynamic>,
      ),
    );

Map<String, dynamic> _$DailyNewspaperToJson(DailyNewspaper instance) =>
    <String, dynamic>{
      'date': instance.date,
      'market_summary': instance.marketSummary,
      'total_opportunities': instance.totalOpportunities,
      'high_confidence_count': instance.highConfidenceCount,
      'avg_iv_rank': instance.avgIvRank,
      'signals': instance.signals,
      'market_conditions': instance.marketConditions,
    };
