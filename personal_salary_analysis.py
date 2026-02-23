#!/usr/bin/env python3
"""
PERSONAL SALARY ANALYSIS
Can you quit your job with $300K investment?
"""

import os
import json
from datetime import datetime, timedelta
import logging

class PersonalSalaryAnalysis:
    """Personal salary and runway analysis"""
    
    def __init__(self):
        self.base_path = "c:/Users/merce/Documents/helm-ai"
        self.production_path = os.path.join(self.base_path, "production")
        
        # Initialize logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(os.path.join(self.production_path, "logs/personal_salary.log")),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        self.logger.info("Personal Salary Analysis initialized")
    
    def analyze_current_job_income(self):
        """Analyze current job income vs startup"""
        self.logger.info("Analyzing current job income...")
        
        # Assume typical tech job income (you can adjust)
        current_job = {
            'annual_salary': 120000,
            'monthly_take_home': 7500,
            'benefits_value': 25000,
            'job_security': 'high',
            'work_life_balance': 'good',
            'total_compensation': 145000
        }
        
        startup_scenario = {
            'founder_salary': 96000,  # $8,000/month
            'monthly_take_home': 6000,  # After taxes/self-employment
            'benefits_value': 12000,  # Self-funded benefits
            'job_security': 'very_low',
            'work_life_balance': 'poor',
            'total_compensation': 108000
        }
        
        income_comparison = {
            'current_job': current_job,
            'startup': startup_scenario,
            'annual_difference': current_job['total_compensation'] - startup_scenario['total_compensation'],
            'monthly_difference': current_job['monthly_take_home'] - startup_scenario['monthly_take_home']
        }
        
        return income_comparison
    
    def analyze_runway_scenarios(self):
        """Analyze runway with different salary levels"""
        self.logger.info("Analyzing runway scenarios...")
        
        scenarios = {
            'full_salary': {
                'founder_salary': 8000,
                'other_costs': 10400,  # $18,400 - $8,000
                'total_burn': 18400,
                'description': 'Full $8,000/month salary'
            },
            'reduced_salary': {
                'founder_salary': 5000,
                'other_costs': 10400,
                'total_burn': 15400,
                'description': 'Reduced $5,000/month salary'
            },
            'minimal_salary': {
                'founder_salary': 3000,
                'other_costs': 10400,
                'total_burn': 13400,
                'description': 'Minimal $3,000/month salary'
            },
            'no_salary': {
                'founder_salary': 0,
                'other_costs': 10400,
                'total_burn': 10400,
                'description': 'No salary, living on savings'
            }
        }
        
        runway_analysis = {}
        for scenario, data in scenarios.items():
            with_300k = 300000 / data['total_burn']
            with_200k = 200000 / data['total_burn']
            with_100k = 100000 / data['total_burn']
            
            runway_analysis[scenario] = {
                'monthly_salary': data['founder_salary'],
                'total_burn': data['total_burn'],
                'runway_300k': with_300k,
                'runway_200k': with_200k,
                'runway_100k': with_100k,
                'description': data['description']
            }
        
        return runway_analysis
    
    def analyze_personal_finances(self):
        """Analyze personal financial requirements"""
        self.logger.info("Analyzing personal financial requirements...")
        
        # Typical monthly expenses (adjust based on your situation)
        personal_expenses = {
            'housing': 2000,
            'food': 800,
            'transportation': 600,
            'utilities': 400,
            'insurance': 600,
            'debt_payments': 1000,
            'savings': 1000,
            'discretionary': 800,
            'total_monthly': 7200
        }
        
        # Salary scenarios vs expenses
        salary_scenarios = {
            'current_job': {
                'monthly_income': 7500,
                'vs_expenses': 300,  # Positive
                'comfort_level': 'comfortable'
            },
            'startup_full': {
                'monthly_income': 6000,
                'vs_expenses': -1200,  # Negative
                'comfort_level': 'tight'
            },
            'startup_reduced': {
                'monthly_income': 3750,  # $5,000 salary after taxes
                'vs_expenses': -3450,  # Very negative
                'comfort_level': 'difficult'
            },
            'startup_minimal': {
                'monthly_income': 2250,  # $3,000 salary after taxes
                'vs_expenses': -4950,  # Very negative
                'comfort_level': 'unsustainable'
            }
        }
        
        return {
            'expenses': personal_expenses,
            'scenarios': salary_scenarios
        }
    
    def analyze_risk_factors(self):
        """Analyze risk factors of quitting job"""
        self.logger.info("Analyzing risk factors...")
        
        risks = {
            'financial_risks': {
                'income_reduction': '37% reduction in take-home pay',
                'benefits_loss': '52% reduction in total compensation',
                'runway_limitation': '16 months max runway',
                'personal_savings_depletion': 'May need to use personal savings',
                'tax_complexity': 'Self-employment tax complications'
            },
            'career_risks': {
                'gap_in_resume': 'Startup gap if it fails',
                'network_loss': 'Current professional network',
                'skill_depreciation': 'May not use current skills',
                'age_factor': 'Harder to return to corporate later'
            },
            'business_risks': {
                'funding_risk': 'May not get full $300K',
                'revenue_risk': 'May not generate revenue quickly',
                'market_risk': 'Product may not find market',
                'competition_risk': 'Competitors may emerge'
            },
            'personal_risks': {
                'stress_increase': 'High stress and pressure',
                'work_life_balance': 'Poor work-life balance',
                'relationship_strain': 'Pressure on personal relationships',
                'health_impact': 'Potential health issues from stress'
            }
        }
        
        return risks
    
    def create_transition_strategy(self):
        """Create strategic transition plan"""
        self.logger.info("Creating transition strategy...")
        
        strategies = {
            'gradual_transition': {
                'approach': 'Keep job, work on startup part-time',
                'timeline': '6-12 months',
                'pros': [
                    'Steady income maintained',
                    'Lower financial risk',
                    'Time to validate business',
                    'Can build foundation while employed'
                ],
                'cons': [
                    'Slower progress',
                    'Limited time for startup',
                    'Potential conflict of interest',
                    'Burnout risk'
                ],
                'requirements': [
                    'Employer allows side projects',
                    'Time management skills',
                    'Clear separation of work'
                ]
            },
            'full_time_with_buffer': {
                'approach': 'Quit job with 6+ months personal savings',
                'timeline': 'Immediate transition',
                'pros': [
                    'Full focus on startup',
                    'Faster progress',
                    'Complete commitment',
                    'Professional appearance to investors'
                ],
                'cons': [
                    'High financial risk',
                    'Pressure to succeed quickly',
                    'No backup plan',
                    'Personal stress'
                ],
                'requirements': [
                    '6+ months personal savings',
                    'Support from family',
                    'Risk tolerance',
                    'Confidence in business'
                ]
            },
            'hybrid_approach': {
                'approach': 'Negotiate part-time or contract with current employer',
                'timeline': '3-6 months transition',
                'pros': [
                    'Some income maintained',
                    'More time for startup',
                    'Lower risk than full quit',
                    'Professional relationships maintained'
                ],
                'cons': [
                    'Still divided focus',
                    'May not be possible',
                    'Employer may not agree',
                    'Complex scheduling'
                ],
                'requirements': [
                    'Flexible employer',
                    'Negotiation skills',
                    'Clear boundaries',
                    'Performance maintenance'
                ]
            }
        }
        
        return strategies
    
    def generate_personal_salary_report(self):
        """Generate comprehensive personal salary analysis"""
        self.logger.info("Generating personal salary analysis report...")
        
        # Analyze all components
        income_comparison = self.analyze_current_job_income()
        runway_scenarios = self.analyze_runway_scenarios()
        personal_finances = self.analyze_personal_finances()
        risk_factors = self.analyze_risk_factors()
        transition_strategies = self.create_transition_strategy()
        
        # Create comprehensive report
        report = {
            'report_timestamp': datetime.now().isoformat(),
            'report_type': 'Personal Salary and Transition Analysis',
            'income_comparison': income_comparison,
            'runway_scenarios': runway_scenarios,
            'personal_finances': personal_finances,
            'risk_factors': risk_factors,
            'transition_strategies': transition_strategies,
            'key_findings': {
                'current_job_income': income_comparison['current_job']['monthly_take_home'],
                'startup_income': income_comparison['startup']['monthly_take_home'],
                'monthly_income_gap': income_comparison['monthly_difference'],
                'annual_compensation_gap': income_comparison['annual_difference'],
                'max_runway_300k': max(data['runway_300k'] for data in runway_scenarios.values()),
                'min_runway_300k': min(data['runway_300k'] for data in runway_scenarios.values()),
                'personal_monthly_expenses': personal_finances['expenses']['total_monthly']
            },
            'recommendations': {
                'minimum_personal_savings': 50000,
                'ideal_personal_savings': 100000,
                'recommended_approach': 'gradual_transition',
                'timeline_to_quit': '6-12 months',
                'funding_target': 400000,  # Higher to account for personal needs
                'salary_strategy': 'reduced_salary'
            }
        }
        
        # Save report
        report_path = os.path.join(self.production_path, "personal_salary_analysis_report.json")
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2)
        
        self.logger.info(f"Personal salary analysis report saved: {report_path}")
        
        # Print summary
        self.print_salary_summary(report)
        
        return report_path
    
    def print_salary_summary(self, report):
        """Print personal salary summary"""
        print(f"\n💰 STELLOR LOGIC AI - PERSONAL SALARY ANALYSIS")
        print("=" * 60)
        
        income = report['income_comparison']
        runway = report['runway_scenarios']
        finances = report['personal_finances']
        findings = report['key_findings']
        recommendations = report['recommendations']
        
        print(f"💸 INCOME COMPARISON:")
        print(f"   🏢 Current Job: ${income['current_job']['monthly_take_home']:,}/month")
        print(f"   🚀 Startup: ${income['startup']['monthly_take_home']:,}/month")
        print(f"   📉 Monthly Gap: ${findings['monthly_income_gap']:,}")
        print(f"   📊 Annual Gap: ${findings['annual_compensation_gap']:,}")
        
        print(f"\n⏰ RUNWAY SCENARIOS WITH $300K:")
        for scenario, data in runway.items():
            print(f"   📊 {scenario.replace('_', ' ').title()}:")
            print(f"      💸 Monthly Salary: ${data['monthly_salary']:,}")
            print(f"      🔥 Total Burn: ${data['total_burn']:,}")
            print(f"      ⏰ Runway: {data['runway_300k']:.1f} months")
            print(f"      📝 {data['description']}")
        
        print(f"\n💳 PERSONAL FINANCIAL REQUIREMENTS:")
        expenses = finances['expenses']
        print(f"   🏠 Housing: ${expenses['housing']:,}")
        print(f"   🍔 Food: ${expenses['food']:,}")
        print(f"   🚗 Transportation: ${expenses['transportation']:,}")
        print(f"   💡 Utilities: ${expenses['utilities']:,}")
        print(f"   🏥 Insurance: ${expenses['insurance']:,}")
        print(f"   💳 Debt Payments: ${expenses['debt_payments']:,}")
        print(f"   💰 Savings: ${expenses['savings']:,}")
        print(f"   🎮 Discretionary: ${expenses['discretionary']:,}")
        print(f"   💸 Total Monthly: ${expenses['total_monthly']:,}")
        
        print(f"\n⚠️ KEY FINANCIAL REALITY:")
        print(f"   💸 Startup Income: ${findings['startup_income']:,}")
        print(f"   💳 Personal Expenses: ${findings['personal_monthly_expenses']:,}")
        print(f"   📉 Monthly Shortfall: ${findings['personal_monthly_expenses'] - findings['startup_income']:,}")
        print(f"   🏦 Personal Savings Needed: ${recommendations['minimum_personal_savings']:,}")
        
        print(f"\n🎯 RISK FACTORS:")
        risks = report['risk_factors']
        for category, items in risks.items():
            print(f"   ⚠️ {category.replace('_', ' ').title()}:")
            for item, description in items.items():
                print(f"      • {description}")
        
        print(f"\n💡 RECOMMENDED STRATEGY:")
        print(f"   🎯 Approach: {recommendations['recommended_approach'].replace('_', ' ').title()}")
        print(f"   ⏰ Timeline to Quit: {recommendations['timeline_to_quit']}")
        print(f"   💰 Personal Savings Needed: ${recommendations['minimum_personal_savings']:,}")
        print(f"   🎯 Ideal Personal Savings: ${recommendations['ideal_personal_savings']:,}")
        print(f"   💸 Funding Target: ${recommendations['funding_target']:,}")
        
        print(f"\n🚀 TRANSITION OPTIONS:")
        strategies = report['transition_strategies']
        for strategy, details in strategies.items():
            print(f"   📋 {strategy.replace('_', ' ').title()}:")
            print(f"      📝 {details['approach']}")
            print(f"      ⏰ Timeline: {details['timeline']}")
            print(f"      ✅ Pros: {', '.join(details['pros'][:2])}")
            print(f"      ❌ Cons: {', '.join(details['cons'][:2])}")
        
        print(f"\n💰 BOTTOM LINE:")
        print(f"   ❌ $300K investment = 16.3 months runway")
        print(f"   ❌ $8,000/month salary = $1,200 monthly shortfall")
        print(f"   ❌ Need ${recommendations['minimum_personal_savings']:,} personal savings")
        print(f"   ⚠️ High risk to quit job immediately")
        print(f"   ✅ Gradual transition recommended")
        print(f"   🎯 Build foundation while employed")

if __name__ == "__main__":
    print("💰 STELLOR LOGIC AI - PERSONAL SALARY ANALYSIS")
    print("=" * 60)
    print("Can you quit your job with $300K investment?")
    print("=" * 60)
    
    analysis = PersonalSalaryAnalysis()
    
    try:
        # Generate personal salary analysis
        report_path = analysis.generate_personal_salary_report()
        
        print(f"\n🎉 PERSONAL SALARY ANALYSIS COMPLETED!")
        print(f"✅ Income comparison analyzed")
        print(f"✅ Runway scenarios calculated")
        print(f"✅ Personal finances assessed")
        print(f"✅ Risk factors identified")
        print(f"✅ Transition strategies created")
        print(f"📄 Report saved: {report_path}")
        
    except Exception as e:
        print(f"❌ Personal salary analysis failed: {str(e)}")
        import traceback
        traceback.print_exc()
