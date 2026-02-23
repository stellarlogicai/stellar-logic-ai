#!/usr/bin/env python3
"""
BETA LAUNCH FINANCIALS
Analyze financial impact of beta launch strategy
"""

import os
import json
from datetime import datetime, timedelta
import logging

class BetaLaunchFinancials:
    """Beta launch financial analysis"""
    
    def __init__(self):
        self.base_path = "c:/Users/merce/Documents/helm-ai"
        self.production_path = os.path.join(self.base_path, "production")
        
        # Initialize logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(os.path.join(self.production_path, "logs/beta_financials.log")),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        # Current cost structure
        self.monthly_costs = {
            'infrastructure': {
                'production_server': 500,
                'monitoring_tools': 200,
                'security_services': 300,
                'backup_storage': 100,
                'total': 1100
            },
            'software_licenses': {
                'development_tools': 300,
                'security_tools': 250,
                'monitoring_licenses': 150,
                'total': 700
            },
            'personnel': {
                'founder_salary': 8000,
                'part_time_developer': 4000,
                'customer_support': 2000,
                'total': 14000
            },
            'operations': {
                'legal_compliance': 500,
                'accounting': 300,
                'marketing': 1000,
                'office_overhead': 800,
                'total': 2600
            }
        }
        
        self.logger.info("Beta Launch Financials initialized")
    
    def calculate_monthly_burn(self):
        """Calculate total monthly burn rate"""
        total_infrastructure = self.monthly_costs['infrastructure']['total']
        total_software = self.monthly_costs['software_licenses']['total']
        total_personnel = self.monthly_costs['personnel']['total']
        total_operations = self.monthly_costs['operations']['total']
        
        total_monthly_burn = total_infrastructure + total_software + total_personnel + total_operations
        
        return {
            'infrastructure': total_infrastructure,
            'software': total_software,
            'personnel': total_personnel,
            'operations': total_operations,
            'total_burn': total_monthly_burn
        }
    
    def analyze_beta_revenue_scenarios(self):
        """Analyze different beta revenue scenarios"""
        scenarios = {
            'conservative': {
                'beta_customers': 5,
                'avg_price': 500,
                'conversion_rate': 0.2,
                'description': 'Conservative beta adoption'
            },
            'moderate': {
                'beta_customers': 10,
                'avg_price': 750,
                'conversion_rate': 0.3,
                'description': 'Moderate beta adoption'
            },
            'aggressive': {
                'beta_customers': 20,
                'avg_price': 1000,
                'conversion_rate': 0.4,
                'description': 'Aggressive beta adoption'
            }
        }
        
        revenue_analysis = {}
        for scenario, data in scenarios.items():
            beta_revenue = data['beta_customers'] * data['avg_price']
            conversion_revenue = beta_revenue * data['conversion_rate']
            total_revenue = beta_revenue + conversion_revenue
            
            revenue_analysis[scenario] = {
                'beta_customers': data['beta_customers'],
                'avg_price': data['avg_price'],
                'conversion_rate': data['conversion_rate'],
                'beta_revenue': beta_revenue,
                'conversion_revenue': conversion_revenue,
                'total_revenue': total_revenue,
                'description': data['description']
            }
        
        return revenue_analysis
    
    def calculate_runway_scenarios(self, investment_amount):
        """Calculate runway scenarios with different investment amounts"""
        monthly_burn = self.calculate_monthly_burn()
        revenue_scenarios = self.analyze_beta_revenue_scenarios()
        
        runway_analysis = {}
        for scenario, revenue_data in revenue_scenarios.items():
            net_burn = monthly_burn['total_burn'] - revenue_data['total_revenue']
            
            if net_burn <= 0:
                runway_months = float('inf')  # Profitable
                cash_position = 'profitable'
            else:
                runway_months = investment_amount / net_burn
                cash_position = 'burning'
            
            runway_analysis[scenario] = {
                'monthly_revenue': revenue_data['total_revenue'],
                'net_burn': net_burn,
                'runway_months': runway_months,
                'cash_position': cash_position,
                'break_even_point': 'immediate' if net_burn <= 0 else f"{runway_months:.1f} months"
            }
        
        return runway_analysis
    
    def analyze_funding_requirements(self):
        """Analyze funding requirements for different scenarios"""
        monthly_burn = self.calculate_monthly_burn()
        revenue_scenarios = self.analyze_beta_revenue_scenarios()
        
        funding_analysis = {}
        
        # Different funding amounts
        funding_amounts = [100000, 200000, 300000, 500000]
        
        for amount in funding_amounts:
            scenario_analysis = {}
            for scenario, revenue_data in revenue_scenarios.items():
                net_burn = monthly_burn['total_burn'] - revenue_data['total_revenue']
                
                if net_burn <= 0:
                    runway = float('inf')
                    additional_funding_needed = 0
                else:
                    runway = amount / net_burn
                    additional_funding_needed = net_burn * 12  # 12 months runway
                
                scenario_analysis[scenario] = {
                    'runway_months': runway,
                    'additional_funding_needed': additional_funding_needed,
                    'profitable': net_burn <= 0
                }
            
            funding_analysis[f'${amount:,}'] = scenario_analysis
        
        return funding_analysis
    
    def generate_beta_financial_report(self):
        """Generate comprehensive beta launch financial report"""
        self.logger.info("Generating beta launch financial report...")
        
        # Calculate all scenarios
        monthly_burn = self.calculate_monthly_burn()
        revenue_scenarios = self.analyze_beta_revenue_scenarios()
        funding_requirements = self.analyze_funding_requirements()
        
        # Calculate with $300K investment
        runway_300k = self.calculate_runway_scenarios(300000)
        
        # Create comprehensive report
        report = {
            'report_timestamp': datetime.now().isoformat(),
            'report_type': 'Beta Launch Financial Analysis',
            'monthly_burn_analysis': monthly_burn,
            'revenue_scenarios': revenue_scenarios,
            'runway_with_300k': runway_300k,
            'funding_requirements': funding_requirements,
            'key_insights': {
                'total_monthly_burn': monthly_burn['total_burn'],
                'break_even_revenue_needed': monthly_burn['total_burn'],
                'conservative_runway_300k': runway_300k['conservative']['runway_months'],
                'moderate_runway_300k': runway_300k['moderate']['runway_months'],
                'aggressive_runway_300k': runway_300k['aggressive']['runway_months'],
                'profitable_scenarios': [
                    scenario for scenario, data in runway_300k.items() 
                    if data['cash_position'] == 'profitable'
                ]
            },
            'recommendations': {
                'minimum_funding': 200000,
                'optimal_funding': 300000,
                'target_beta_customers': 15,
                'target_avg_price': 750,
                'break_even_timeline': '6-9 months'
            }
        }
        
        # Save report
        report_path = os.path.join(self.production_path, "beta_launch_financials_report.json")
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2)
        
        self.logger.info(f"Beta launch financials report saved: {report_path}")
        
        # Print summary
        self.print_financial_summary(report)
        
        return report_path
    
    def print_financial_summary(self, report):
        """Print financial summary"""
        print(f"\n💰 STELLOR LOGIC AI - BETA LAUNCH FINANCIAL ANALYSIS")
        print("=" * 60)
        
        burn = report['monthly_burn_analysis']
        scenarios = report['revenue_scenarios']
        runway = report['runway_with_300k']
        insights = report['key_insights']
        recommendations = report['recommendations']
        
        print(f"💸 MONTHLY BURN RATE:")
        print(f"   🏗️ Infrastructure: ${burn['infrastructure']:,}")
        print(f"   💻 Software Licenses: ${burn['software']:,}")
        print(f"   👥 Personnel: ${burn['personnel']:,}")
        print(f"   🏢 Operations: ${burn['operations']:,}")
        print(f"   💰 TOTAL BURN: ${burn['total_burn']:,}/month")
        
        print(f"\n📈 BETA REVENUE SCENARIOS:")
        for scenario, data in scenarios.items():
            print(f"   📊 {scenario.title()}:")
            print(f"      👥 Beta Customers: {data['beta_customers']}")
            print(f"      💸 Avg Price: ${data['avg_price']:,}")
            print(f"      📈 Monthly Revenue: ${data['total_revenue']:,}")
            print(f"      📝 {data['description']}")
        
        print(f"\n⏰ RUNWAY WITH $300K INVESTMENT:")
        for scenario, data in runway.items():
            if data['runway_months'] == float('inf'):
                print(f"   📊 {scenario.title()}: ✅ PROFITABLE (${data['monthly_revenue']:,} revenue)")
            else:
                print(f"   📊 {scenario.title()}: {data['runway_months']:.1f} months")
                print(f"      💸 Net Burn: ${data['net_burn']:,}/month")
        
        print(f"\n🎯 KEY INSIGHTS:")
        print(f"   💸 Total Monthly Burn: ${insights['total_monthly_burn']:,}")
        print(f"   📈 Break-even Revenue Needed: ${insights['break_even_revenue_needed']:,}")
        print(f"   ⏰ Conservative Runway: {insights['conservative_runway_300k']:.1f} months")
        print(f"   ⏰ Moderate Runway: {insights['moderate_runway_300k']:.1f} months")
        print(f"   ⏰ Aggressive Runway: {insights['aggressive_runway_300k']:.1f} months")
        
        if insights['profitable_scenarios']:
            print(f"   ✅ Profitable Scenarios: {', '.join(insights['profitable_scenarios'])}")
        else:
            print(f"   ❌ No scenarios profitable initially")
        
        print(f"\n💡 RECOMMENDATIONS:")
        print(f"   💰 Minimum Funding: ${recommendations['minimum_funding']:,}")
        print(f"   🎯 Optimal Funding: ${recommendations['optimal_funding']:,}")
        print(f"   👥 Target Beta Customers: {recommendations['target_beta_customers']}")
        print(f"   💸 Target Average Price: ${recommendations['target_avg_price']:,}")
        print(f"   ⏰ Break-even Timeline: {recommendations['break_even_timeline']}")
        
        # Bottom line assessment
        moderate_runway = insights['moderate_runway_300k']
        if moderate_runway >= 12:
            print(f"\n✅ BOTTOM LINE: $300K provides {moderate_runway:.1f} months runway")
            print(f"   🎯 Good for beta launch and market validation")
        elif moderate_runway >= 6:
            print(f"\n⚠️ BOTTOM LINE: $300K provides {moderate_runway:.1f} months runway")
            print(f"   🎯 Tight but manageable for beta launch")
        else:
            print(f"\n❌ BOTTOM LINE: $300K provides only {moderate_runway:.1f} months runway")
            print(f"   🎯 May need additional funding or cost reduction")

if __name__ == "__main__":
    print("💰 STELLOR LOGIC AI - BETA LAUNCH FINANCIAL ANALYSIS")
    print("=" * 60)
    print("Analyzing financial impact of beta launch strategy")
    print("=" * 60)
    
    financials = BetaLaunchFinancials()
    
    try:
        # Generate financial analysis
        report_path = financials.generate_beta_financial_report()
        
        print(f"\n🎉 BETA LAUNCH FINANCIALS COMPLETED!")
        print(f"✅ Monthly burn rate calculated")
        print(f"✅ Revenue scenarios analyzed")
        print(f"✅ Runway projections created")
        print(f"✅ Funding requirements assessed")
        print(f"📄 Report saved: {report_path}")
        
    except Exception as e:
        print(f"❌ Financial analysis failed: {str(e)}")
        import traceback
        traceback.print_exc()
