from .models import *
from rest_framework import serializers

class TransactionPredictionSerializer(serializers.Serializer):
    transfer_time = serializers.DateTimeField(help_text='Transaction time', required=True)
    amount = serializers.FloatField(help_text='Transaction amount', required=True)
    withdrawable_cash = serializers.FloatField(help_text='Available cash for withdrawal', required=True)
    hesap_acilis_tarihi = serializers.DateField(help_text='Account opening date', required=True)
    uyruk = serializers.CharField(help_text='Nationality', required=True)
    hesap_acilis_tipi = serializers.CharField(help_text='Account opening type', required=True)
    yas = serializers.IntegerField(help_text='Age', required=True)
    meslek = serializers.CharField(help_text='Occupation', required=True)
    bist_tl_cinsinden_hacim = serializers.FloatField(help_text='BIST volume in TL', required=True)
    us_borsasi_usd_cinsinden_hacim = serializers.FloatField(help_text='US market volume in USD', required=True)
    usd_toplam_islem_hacmi = serializers.FloatField(help_text='Total USD transaction volume', required=True)
    ikamet_ili = serializers.FloatField(help_text='City code', required=True)
    farkli_kisi_deposit_amount_try = serializers.FloatField(help_text='Deposit amount from different people in TRY', required=True, allow_null=True)
    farkli_kisi_sayisi = serializers.FloatField(help_text='Number of different people', required=True, allow_null=True)
