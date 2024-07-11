from django.db import models
import re


class Request(models.Model):
    ASSET_CHOICES = [
        ('BTC', 'Bitcoin'),
        ('ETH', 'Ethereum'),
        ('XRP', 'Ripple'),
        # Add more crypto assets as needed
    ]

    assets = models.CharField(max_length=255, choices=ASSET_CHOICES)
    interval = models.CharField(max_length=10)
    look_back_period = models.CharField(max_length=10)
    investment_amount = models.DecimalField(max_digits=10, decimal_places=2)

    def clean(self):
        if not re.match(r"(\d+)([smhdMy])", self.interval):
            raise ValueError('Invalid interval format')

        if not re.match(r"(\d+)([smhdMy])", self.look_back_period):
            raise ValueError('Invalid look back period format')
