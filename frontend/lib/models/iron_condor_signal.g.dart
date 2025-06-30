// GENERATED CODE - DO NOT MODIFY BY HAND

part of 'iron_condor_signal.dart';

// **************************************************************************
// JsonSerializableGenerator
// **************************************************************************

IronCondorSignal _$IronCondorSignalFromJson(Map<String, dynamic> json) =>
    IronCondorSignal(
      ticker: json['ticker'] as String,
      currentPrice: (json['current_price'] as num).toDouble(),
      putShortStrike: (json['put_short_strike'] as num).toDouble(),
      putLongStrike: (json['put_long_strike'] as num).toDouble(),
      callShortStrike: (json['call_short_strike'] as num).toDouble(),
      callLongStrike: (json['call_long_strike'] as num).toDouble(),
      ivRank: (json['iv_rank'] as num).toDouble(),
      premiumCollected: (json['premium_collected'] as num).toDouble(),
      maxProfit: (json['max_profit'] as num).toDouble(),
      maxLoss: (json['max_loss'] as num).toDouble(),
      breakEvenLower: (json['break_even_lower'] as num).toDouble(),
      breakEvenUpper: (json['break_even_upper'] as num).toDouble(),
      confidenceScore: (json['confidence_score'] as num).toDouble(),
      volume: (json['volume'] as num).toInt(),
      expirationDate: json['expiration_date'] as String,
    );

Map<String, dynamic> _$IronCondorSignalToJson(IronCondorSignal instance) =>
    <String, dynamic>{
      'ticker': instance.ticker,
      'current_price': instance.currentPrice,
      'put_short_strike': instance.putShortStrike,
      'put_long_strike': instance.putLongStrike,
      'call_short_strike': instance.callShortStrike,
      'call_long_strike': instance.callLongStrike,
      'iv_rank': instance.ivRank,
      'premium_collected': instance.premiumCollected,
      'max_profit': instance.maxProfit,
      'max_loss': instance.maxLoss,
      'break_even_lower': instance.breakEvenLower,
      'break_even_upper': instance.breakEvenUpper,
      'confidence_score': instance.confidenceScore,
      'volume': instance.volume,
      'expiration_date': instance.expirationDate,
    };
