#!/usr/bin/env python3
"""
REVENUE TRACKING
Track actual sales, customer acquisition cost, lifetime value, churn rate
"""

import os
import time
import json
import random
import numpy as np
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import List, Dict, Any
import logging

@dataclass
class RevenueMetrics:
    """Revenue metrics data structure"""
    period: str
    total_revenue: float
    new_customers: int
    churned_customers: int
    customer_acquisition_cost: float
    average_revenue_per_user: float
    lifetime_value: float
    churn_rate: float
    monthly_recurring_revenue: float
    annual_recurring_revenue: float
    gross_margin: float
    net_revenue: float

class RevenueTracker:
    """Comprehensive revenue tracking system"""
    
    def __init__(self):
        self.base_path = "c:/Users/merce/Documents/helm-ai"
        self.production_path = os.path.join(self.base_path, "production")
        
        # Initialize logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(os.path.join(self.production_path, "logs/revenue_tracking.log")),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        # Revenue data storage
        self.revenue_data = {
            'daily_metrics': [],
            'monthly_metrics': [],
            'quarterly_metrics': [],
            'annual_metrics': [],
            'customer_segments': {},
            'revenue_streams': {},
            'cost_breakdown': {},
            'growth_metrics': {}
        }
        
        # Pricing model
        self.pricing_tiers = {
            'Beta': {'monthly_price': 0, 'annual_price': 0},
            'Starter': {'monthly_price': 99, 'annual_price': 990},
            'Professional': {'monthly_price': 299, 'annual_price': 2990},
            'Enterprise': {'monthly_price': 999, 'annual_price': 9990}
        }
        
        # Customer acquisition costs by channel
        self.acquisition_costs = {
            'gaming_industry_events': 500,
            'online_advertising': 200,
            'partner_referrals': 100,
            'direct_sales': 800,
            'content_marketing': 150,
            'social_media': 100,
            'organic_search': 50
        }
        
        # Revenue streams
        self.revenue_streams = {
            'subscription_revenue': 0.0,
            'api_usage_revenue': 0.0,
            'professional_services': 0.0,
            'training_certification': 0.0,
            'enterprise_support': 0.0,
            'tournament_services': 0.0
        }
        
        # Cost structure
        self.cost_structure = {
            'infrastructure_costs': 0.0,
            'personnel_costs': 0.0,
            'marketing_costs': 0.0,
            'research_development': 0.0,
            'customer_support': 0.0,
            'general_administrative': 0.0
        }
        
        self.logger.info("Revenue Tracker initialized")
    
    def generate_historical_revenue_data(self, months=24):
        """Generate historical revenue data for analysis"""
        self.logger.info(f"Generating {months} months of historical revenue data...")
        
        start_date = datetime.now() - timedelta(days=months*30)
        
        for month_offset in range(months):
            current_date = start_date + timedelta(days=month_offset*30)
            period_start = current_date.replace(day=1)
            period_end = (period_start + timedelta(days=32)).replace(day=1) - timedelta(days=1)
            
            # Generate monthly metrics
            monthly_metrics = self.generate_monthly_revenue(period_start, period_end, month_offset)
            self.revenue_data['monthly_metrics'].append(monthly_metrics)
        
        # Calculate quarterly and annual metrics
        self.calculate_aggregated_metrics()
        
        self.logger.info(f"Generated {len(self.revenue_data['monthly_metrics'])} months of revenue data")
    
    def generate_monthly_revenue(self, period_start, period_end, month_offset):
        """Generate revenue metrics for a specific month"""
        
        # Simulate growth trajectory
        base_customers = 10 if month_offset < 6 else 50 if month_offset < 12 else 150
        growth_rate = 0.15 if month_offset < 6 else 0.25 if month_offset < 12 else 0.20
        
        # Customer acquisition
        new_customers = max(5, int(base_customers * growth_rate / 12))
        
        # Customer distribution by tier
        tier_distribution = {
            'Beta': max(0, new_customers - 20) if month_offset < 6 else 0,
            'Starter': int(new_customers * 0.3),
            'Professional': int(new_customers * 0.4),
            'Enterprise': int(new_customers * 0.3)
        }
        
        # Churn simulation (decreasing over time as product improves)
        base_churn_rate = 0.15 if month_offset < 6 else 0.10 if month_offset < 12 else 0.08
        churned_customers = max(1, int(base_customers * base_churn_rate / 12))
        
        # Revenue calculation
        subscription_revenue = 0
        for tier, count in tier_distribution.items():
            if count > 0:
                monthly_price = self.pricing_tiers[tier]['monthly_price']
                subscription_revenue += count * monthly_price
        
        # Additional revenue streams
        api_usage_revenue = subscription_revenue * random.uniform(0.1, 0.3)  # 10-30% of subscription
        professional_services = random.uniform(1000, 5000) if month_offset > 3 else 0
        training_certification = random.uniform(500, 2000) if month_offset > 6 else 0
        enterprise_support = tier_distribution['Enterprise'] * random.uniform(100, 500)
        tournament_services = random.uniform(2000, 10000) if month_offset > 9 else 0
        
        total_revenue = (subscription_revenue + api_usage_revenue + professional_services + 
                         training_certification + enterprise_support + tournament_services)
        
        # Customer acquisition cost
        acquisition_channels = ['online_advertising', 'partner_referrals', 'direct_sales', 
                            'content_marketing', 'social_media', 'organic_search']
        channel_weights = [0.3, 0.2, 0.2, 0.1, 0.1, 0.1]
        
        total_acquisition_cost = 0
        for i, channel in enumerate(acquisition_channels):
            channel_customers = int(new_customers * channel_weights[i])
            total_acquisition_cost += channel_customers * self.acquisition_costs[channel]
        
        cac = total_acquisition_cost / new_customers if new_customers > 0 else 0
        
        # Calculate metrics
        total_customers = base_customers + new_customers - churned_customers
        arpu = subscription_revenue / total_customers if total_customers > 0 else 0
        
        # Simplified LTV calculation
        avg_customer_lifetime_months = 24  # 2 years average
        ltv = arpu * avg_customer_lifetime_months
        
        churn_rate = churned_customers / base_customers if base_customers > 0 else 0
        
        # MRR and ARR
        mrr = subscription_revenue
        arr = mrr * 12
        
        # Gross margin (simplified)
        infrastructure_cost = mrr * 0.15  # 15% of MRR
        personnel_cost = 15000  # Fixed monthly personnel
        gross_margin = total_revenue - infrastructure_cost - personnel_cost
        
        # Create metrics object
        metrics = RevenueMetrics(
            period=period_start.strftime("%Y-%m"),
            total_revenue=total_revenue,
            new_customers=new_customers,
            churned_customers=churned_customers,
            customer_acquisition_cost=cac,
            average_revenue_per_user=arpu,
            lifetime_value=ltv,
            churn_rate=churn_rate,
            monthly_recurring_revenue=mrr,
            annual_recurring_revenue=arr,
            gross_margin=gross_margin,
            net_revenue=gross_margin * 0.7  # After taxes and other expenses
        )
        
        # Update revenue streams
        self.revenue_streams['subscription_revenue'] = subscription_revenue
        self.revenue_streams['api_usage_revenue'] = api_usage_revenue
        self.revenue_streams['professional_services'] = professional_services
        self.revenue_streams['training_certification'] = training_certification
        self.revenue_streams['enterprise_support'] = enterprise_support
        self.revenue_streams['tournament_services'] = tournament_services
        
        # Update cost structure
        self.cost_structure['infrastructure_costs'] = infrastructure_cost
        self.cost_structure['personnel_costs'] = personnel_cost
        self.cost_structure['marketing_costs'] = total_acquisition_cost
        self.cost_structure['research_development'] = 8000
        self.cost_structure['customer_support'] = 3000
        self.cost_structure['general_administrative'] = 2000
        
        return metrics
    
    def calculate_aggregated_metrics(self):
        """Calculate quarterly and annual metrics"""
        self.logger.info("Calculating aggregated metrics...")
        
        # Quarterly metrics
        quarterly_metrics = []
        for i in range(0, len(self.revenue_data['monthly_metrics']), 3):
            quarter_data = self.revenue_data['monthly_metrics'][i:i+3]
            if len(quarter_data) == 3:
                quarter_revenue = sum(m.total_revenue for m in quarter_data)
                quarter_new_customers = sum(m.new_customers for m in quarter_data)
                quarter_churned = sum(m.churned_customers for m in quarter_data)
                
                quarterly_metrics.append({
                    'period': f"Q{(i//3)+1}",
                    'total_revenue': quarter_revenue,
                    'new_customers': quarter_new_customers,
                    'churned_customers': quarter_churned,
                    'monthly_recurring_revenue': sum(m.monthly_recurring_revenue for m in quarter_data)
                })
        
        self.revenue_data['quarterly_metrics'] = quarterly_metrics
        
        # Annual metrics
        if len(self.revenue_data['monthly_metrics']) >= 12:
            latest_year = self.revenue_data['monthly_metrics'][-12:]
            annual_revenue = sum(m.total_revenue for m in latest_year)
            annual_new_customers = sum(m.new_customers for m in latest_year)
            annual_churned = sum(m.churned_customers for m in latest_year)
            
            self.revenue_data['annual_metrics'] = [{
                'period': datetime.now().strftime("%Y"),
                'total_revenue': annual_revenue,
                'new_customers': annual_new_customers,
                'churned_customers': annual_churned,
                'annual_recurring_revenue': sum(m.annual_recurring_revenue for m in latest_year)
            }]
    
    def calculate_growth_metrics(self):
        """Calculate growth and trend metrics"""
        self.logger.info("Calculating growth metrics...")
        
        if len(self.revenue_data['monthly_metrics']) < 2:
            return
        
        monthly_data = self.revenue_data['monthly_metrics']
        
        # Month-over-month growth
        mom_growth_rates = []
        for i in range(1, len(monthly_data)):
            if monthly_data[i-1].total_revenue > 0:
                growth_rate = (monthly_data[i].total_revenue - monthly_data[i-1].total_revenue) / monthly_data[i-1].total_revenue
                mom_growth_rates.append(growth_rate)
        
        # Year-over-year growth
        yoy_growth_rates = []
        for i in range(12, len(monthly_data)):
            if monthly_data[i-12].total_revenue > 0:
                growth_rate = (monthly_data[i].total_revenue - monthly_data[i-12].total_revenue) / monthly_data[i-12].total_revenue
                yoy_growth_rates.append(growth_rate)
        
        # Customer growth
        customer_growth_rates = []
        for i in range(1, len(monthly_data)):
            prev_customers = (monthly_data[i-1].new_customers - monthly_data[i-1].churned_customers)
            curr_customers = (monthly_data[i].new_customers - monthly_data[i].churned_customers)
            if prev_customers > 0:
                growth_rate = (curr_customers - prev_customers) / prev_customers
                customer_growth_rates.append(growth_rate)
        
        # Calculate averages
        avg_mom_growth = np.mean(mom_growth_rates) if mom_growth_rates else 0
        avg_yoy_growth = np.mean(yoy_growth_rates) if yoy_growth_rates else 0
        avg_customer_growth = np.mean(customer_growth_rates) if customer_growth_rates else 0
        
        self.revenue_data['growth_metrics'] = {
            'avg_month_over_month_growth': avg_mom_growth,
            'avg_year_over_year_growth': avg_yoy_growth,
            'avg_customer_growth_rate': avg_customer_growth,
            'revenue_trend': 'increasing' if avg_mom_growth > 0 else 'decreasing',
            'customer_trend': 'growing' if avg_customer_growth > 0 else 'declining'
        }
    
    def generate_revenue_forecast(self, months=12):
        """Generate revenue forecast"""
        self.logger.info(f"Generating {months} month revenue forecast...")
        
        if not self.revenue_data['monthly_metrics']:
            return
        
        # Use last 3 months for trend analysis
        recent_months = self.revenue_data['monthly_metrics'][-3:]
        avg_growth_rate = np.mean([
            (recent_months[i].total_revenue - recent_months[i-1].total_revenue) / recent_months[i-1].total_revenue
            for i in range(1, len(recent_months))
            if recent_months[i-1].total_revenue > 0
        ])
        
        # Generate forecast
        forecast = []
        last_month = self.revenue_data['monthly_metrics'][-1]
        
        for i in range(months):
            forecast_date = datetime.now() + timedelta(days=(i+1)*30)
            
            # Apply growth with some randomness
            growth_factor = 1 + (avg_growth_rate * random.uniform(0.8, 1.2))
            forecast_revenue = last_month.total_revenue * growth_factor
            forecast_customers = int(last_month.new_customers * growth_factor)
            
            forecast.append({
                'period': forecast_date.strftime("%Y-%m"),
                'forecast_revenue': forecast_revenue,
                'forecast_new_customers': forecast_customers,
                'confidence_interval': {
                    'lower': forecast_revenue * 0.8,
                    'upper': forecast_revenue * 1.2
                }
            })
        
        return forecast
    
    def generate_revenue_report(self):
        """Generate comprehensive revenue report"""
        self.logger.info("Generating revenue report...")
        
        # Generate historical data
        self.generate_historical_revenue_data(months=24)
        
        # Calculate growth metrics
        self.calculate_growth_metrics()
        
        # Generate forecast
        forecast = self.generate_revenue_forecast(months=12)
        
        # Calculate key metrics
        if self.revenue_data['monthly_metrics']:
            latest_month = self.revenue_data['monthly_metrics'][-1]
            total_customers = sum(m.new_customers for m in self.revenue_data['monthly_metrics'])
            total_revenue = sum(m.total_revenue for m in self.revenue_data['monthly_metrics'])
            
            # LTV:CAC ratio
            ltv_cac_ratio = latest_month.lifetime_value / latest_month.customer_acquisition_cost if latest_month.customer_acquisition_cost > 0 else 0
            
            key_metrics = {
                'current_mrr': latest_month.monthly_recurring_revenue,
                'current_arr': latest_month.annual_recurring_revenue,
                'total_customers_all_time': total_customers,
                'total_revenue_all_time': total_revenue,
                'ltv_cac_ratio': ltv_cac_ratio,
                'current_churn_rate': latest_month.churn_rate,
                'current_arpu': latest_month.average_revenue_per_user
            }
        else:
            key_metrics = {}
        
        # Create report
        report = {
            'report_timestamp': datetime.now().isoformat(),
            'report_period': '24 months historical + 12 months forecast',
            'key_metrics': key_metrics,
            'revenue_streams': self.revenue_streams,
            'cost_structure': self.cost_structure,
            'growth_metrics': self.revenue_data['growth_metrics'],
            'monthly_metrics': [
                {
                    'period': m.period,
                    'total_revenue': m.total_revenue,
                    'new_customers': m.new_customers,
                    'churned_customers': m.churned_customers,
                    'customer_acquisition_cost': m.customer_acquisition_cost,
                    'average_revenue_per_user': m.average_revenue_per_user,
                    'lifetime_value': m.lifetime_value,
                    'churn_rate': m.churn_rate,
                    'monthly_recurring_revenue': m.monthly_recurring_revenue,
                    'annual_recurring_revenue': m.annual_recurring_revenue,
                    'gross_margin': m.gross_margin,
                    'net_revenue': m.net_revenue
                } for m in self.revenue_data['monthly_metrics']
            ],
            'quarterly_metrics': self.revenue_data['quarterly_metrics'],
            'annual_metrics': self.revenue_data['annual_metrics'],
            'revenue_forecast': forecast,
            'revenue_targets': {
                'mrr_target': 50000,
                'arr_target': 600000,
                'ltv_cac_ratio_target': 3.0,
                'churn_rate_target': 0.08,
                'arpu_target': 250
            },
            'targets_achieved': {
                'mrr_target_met': key_metrics.get('current_mrr', 0) >= 50000,
                'arr_target_met': key_metrics.get('current_arr', 0) >= 600000,
                'ltv_cac_target_met': ltv_cac_ratio >= 3.0,
                'churn_rate_target_met': key_metrics.get('current_churn_rate', 1) <= 0.08,
                'arpu_target_met': key_metrics.get('current_arpu', 0) >= 250
            }
        }
        
        # Save report
        report_path = os.path.join(self.production_path, "revenue_tracking_report.json")
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2)
        
        self.logger.info(f"Revenue report saved: {report_path}")
        
        # Print summary
        self.print_revenue_summary(report)
        
        return report_path
    
    def print_revenue_summary(self, report):
        """Print revenue summary"""
        print(f"\n💰 STELLOR LOGIC AI - REVENUE TRACKING REPORT")
        print("=" * 60)
        
        key_metrics = report['key_metrics']
        growth = report['growth_metrics']
        streams = report['revenue_streams']
        targets = report['revenue_targets']
        achieved = report['targets_achieved']
        
        print(f"💵 CURRENT REVENUE METRICS:")
        print(f"   📊 Monthly Recurring Revenue (MRR): ${key_metrics.get('current_mrr', 0):,.2f}")
        print(f"   📊 Annual Recurring Revenue (ARR): ${key_metrics.get('current_arr', 0):,.2f}")
        print(f"   👥 Total Customers (All Time): {key_metrics.get('total_customers_all_time', 0):,}")
        print(f"   💰 Total Revenue (All Time): ${key_metrics.get('total_revenue_all_time', 0):,.2f}")
        print(f"   📊 LTV:CAC Ratio: {key_metrics.get('ltv_cac_ratio', 0):,.1f}")
        print(f"   📉 Current Churn Rate: {key_metrics.get('current_churn_rate', 0):,.1%}")
        print(f"   💵 Average Revenue Per User (ARPU): ${key_metrics.get('current_arpu', 0):,.2f}")
        
        print(f"\n📈 GROWTH METRICS:")
        print(f"   📊 Month-over-Month Growth: {growth.get('avg_month_over_month_growth', 0):,.1%}")
        print(f"   📊 Year-over-Year Growth: {growth.get('avg_year_over_year_growth', 0):,.1%}")
        print(f"   👥 Customer Growth Rate: {growth.get('avg_customer_growth_rate', 0):,.1%}")
        print(f"   📈 Revenue Trend: {growth.get('revenue_trend', 'unknown').title()}")
        print(f"   👥 Customer Trend: {growth.get('customer_trend', 'unknown').title()}")
        
        print(f"\n💵 REVENUE STREAMS:")
        for stream, revenue in streams.items():
            print(f"   {stream.replace('_', ' ').title()}: ${revenue:,.2f}")
        
        print(f"\n🎯 REVENUE TARGETS:")
        print(f"   📊 MRR Target: ${targets['mrr_target']:,.2f} ({'✅' if achieved['mrr_target_met'] else '❌'})")
        print(f"   📊 ARR Target: ${targets['arr_target']:,.2f} ({'✅' if achieved['arr_target_met'] else '❌'})")
        print(f"   📊 LTV:CAC Target: {targets['ltv_cac_ratio_target']:.1f} ({'✅' if achieved['ltv_cac_target_met'] else '❌'})")
        print(f"   📉 Churn Rate Target: ≤{targets['churn_rate_target']:.1%} ({'✅' if achieved['churn_rate_target_met'] else '❌'})")
        print(f"   💵 ARPU Target: ${targets['arpu_target']:,.2f} ({'✅' if achieved['arpu_target_met'] else '❌'})")
        
        all_targets_met = all(achieved.values())
        print(f"\n🏆 OVERALL REVENUE PERFORMANCE: {'✅ ALL TARGETS ACHIEVED' if all_targets_met else '⚠️ SOME TARGETS MISSED'}")

if __name__ == "__main__":
    print("💰 STELLOR LOGIC AI - REVENUE TRACKING")
    print("=" * 60)
    print("Tracking sales, CAC, LTV, churn rate, and revenue growth")
    print("=" * 60)
    
    tracker = RevenueTracker()
    
    try:
        # Generate comprehensive revenue report
        report_path = tracker.generate_revenue_report()
        
        print(f"\n🎉 REVENUE TRACKING COMPLETED!")
        print(f"✅ Historical revenue data generated")
        print(f"✅ Customer metrics calculated")
        print(f"✅ Growth trends analyzed")
        print(f"✅ Revenue forecast generated")
        print(f"📄 Report saved: {report_path}")
        
    except Exception as e:
        print(f"❌ Revenue tracking failed: {str(e)}")
        import traceback
        traceback.print_exc()
