import 'package:json_annotation/json_annotation.dart';

part 'iron_condor_signal.g.dart';

@JsonSerializable()
class IronCondorSignal {
  final String ticker;
  @JsonKey(name: 'current_price')
  final double currentPrice;
  @JsonKey(name: 'put_short_strike')
  final double putShortStrike;
  @JsonKey(name: 'put_long_strike')
  final double putLongStrike;
  @JsonKey(name: 'call_short_strike')
  final double callShortStrike;
  @JsonKey(name: 'call_long_strike')
  final double callLongStrike;
  @JsonKey(name: 'iv_rank')
  final double ivRank;
  @JsonKey(name: 'premium_collected')
  final double premiumCollected;
  @JsonKey(name: 'max_profit')
  final double maxProfit;
  @JsonKey(name: 'max_loss')
  final double maxLoss;
  @JsonKey(name: 'break_even_lower')
  final double breakEvenLower;
  @JsonKey(name: 'break_even_upper')
  final double breakEvenUpper;
  @JsonKey(name: 'confidence_score')
  final double confidenceScore;
  final int volume;
  @JsonKey(name: 'expiration_date')
  final String expirationDate;

  IronCondorSignal({
    required this.ticker,
    required this.currentPrice,
    required this.putShortStrike,
    required this.putLongStrike,
    required this.callShortStrike,
    required this.callLongStrike,
    required this.ivRank,
    required this.premiumCollected,
    required this.maxProfit,
    required this.maxLoss,
    required this.breakEvenLower,
    required this.breakEvenUpper,
    required this.confidenceScore,
    required this.volume,
    required this.expirationDate,
  });

  factory IronCondorSignal.fromJson(Map<String, dynamic> json) =>
      _$IronCondorSignalFromJson(json);

  Map<String, dynamic> toJson() => _$IronCondorSignalToJson(this);

  // Helper methods for UI
  String get confidenceLevel {
    if (confidenceScore >= 80) return 'High';
    if (confidenceScore >= 60) return 'Good';
    if (confidenceScore >= 40) return 'Moderate';
    return 'Low';
  }

  String get profitPotential {
    final profitPercent = (maxProfit / maxLoss) * 100;
    if (profitPercent >= 30) return 'Excellent';
    if (profitPercent >= 20) return 'Good';
    if (profitPercent >= 15) return 'Fair';
    return 'Poor';
  }

  double get profitPercentage => (maxProfit / maxLoss) * 100;

  String get volumeFormatted {
    if (volume >= 1000000) {
      return '${(volume / 1000000).toStringAsFixed(1)}M';
    } else if (volume >= 1000) {
      return '${(volume / 1000).toStringAsFixed(0)}K';
    }
    return volume.toString();
  }
}