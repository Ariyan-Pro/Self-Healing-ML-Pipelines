#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# enterprise_platform/enterprise_metrics/executive_dashboard.py
"""
Executive dashboard for enterprise metrics visualization.
"""

import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path


class ExecutiveDashboard:
    """Executive dashboard for visualizing enterprise metrics."""
    
    def __init__(self, metrics_system):
        self.metrics_system = metrics_system
    
    def generate_financial_dashboard(self, tenant_id: str, output_path: str = None):
        """Generate financial metrics dashboard."""
        report = self.metrics_system.generate_executive_report(tenant_id)
        financial = report["financial_metrics"]
        
        # Create figure
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle(f"Financial Dashboard - {tenant_id}", fontsize=16, fontweight='bold')
        
        # 1. Cost Savings Breakdown
        savings_labels = ['Engineer Cost', 'Downtime', 'Infrastructure']
        savings_values = [
            financial["engineer_cost_saved"],
            financial["downtime_cost_saved"],
            financial["infrastructure_savings"]
        ]
        
        axes[0, 0].pie(savings_values, labels=savings_labels, autopct='%1.1f%%', startangle=90)
        axes[0, 0].set_title("Cost Savings Breakdown")
        
        # 2. ROI Trend (simulated)
        months = ['Oct', 'Nov', 'Dec', 'Jan']
        roi_trend = [85, 92, 105, financial.get("estimated_roi", 120)]
        
        axes[0, 1].plot(months, roi_trend, marker='o', linewidth=2)
        axes[0, 1].axhline(y=100, color='r', linestyle='--', alpha=0.5, label='Break-even')
        axes[0, 1].fill_between(months, roi_trend, 100, where=[r >= 100 for r in roi_trend], 
                               alpha=0.3, color='green')
        axes[0, 1].fill_between(months, roi_trend, 100, where=[r < 100 for r in roi_trend], 
                               alpha=0.3, color='red')
        axes[0, 1].set_title("ROI Trend (%)")
        axes[0, 1].set_ylabel("ROI %")
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Monthly Savings
        months_full = ['Sep', 'Oct', 'Nov', 'Dec', 'Jan']
        monthly_savings = [12000, 14500, 16200, 17800, financial["total_cost_savings"]]
        
        bars = axes[1, 0].bar(months_full, monthly_savings, color=['blue', 'blue', 'blue', 'blue', 'green'])
        axes[1, 0].set_title("Monthly Cost Savings ($)")
        axes[1, 0].set_ylabel("USD")
        
        # Add value labels on bars
        for bar, value in zip(bars, monthly_savings):
            height = bar.get_height()
            axes[1, 0].text(bar.get_x() + bar.get_width()/2., height + 500,
                           f'${value:,.0f}', ha='center', va='bottom', fontsize=9)
        
        # 4. Payback Period
        investment = 50000  # Example initial investment
        cumulative_savings = np.cumsum(monthly_savings)
        payback_month = None
        
        for i, cumulative in enumerate(cumulative_savings):
            if cumulative >= investment:
                payback_month = i
                break
        
        axes[1, 1].plot(months_full, cumulative_savings, marker='o', linewidth=2, label='Cumulative Savings')
        axes[1, 1].axhline(y=investment, color='r', linestyle='--', alpha=0.5, label='Initial Investment')
        
        if payback_month is not None:
            axes[1, 1].axvline(x=payback_month, color='g', linestyle=':', alpha=0.7, 
                              label=f'Payback: {months_full[payback_month]}')
        
        axes[1, 1].set_title("Cumulative Savings vs Investment")
        axes[1, 1].set_ylabel("USD")
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"Dashboard saved to {output_path}")
        else:
            plt.show()
    
    def generate_risk_dashboard(self, tenant_id: str, output_path: str = None):
        """Generate risk metrics dashboard."""
        report = self.metrics_system.generate_executive_report(tenant_id)
        risk = report["risk_metrics"]
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle(f"Risk Management Dashboard - {tenant_id}", fontsize=16, fontweight='bold')
        
        # 1. Risk Exposure Gauge
        risk_score = risk["risk_exposure_score"]
        risk_level = risk["risk_level"]
        
        # Create gauge chart
        ax = axes[0, 0]
        self._create_risk_gauge(ax, risk_score, risk_level)
        
        # 2. Incident Severity Distribution
        severity_labels = ['Low', 'Medium', 'High', 'Critical']
        # Simulated data
        severity_counts = [45, 28, 12, risk["catastrophic_failures"]]
        colors = ['green', 'yellow', 'orange', 'red']
        
        bars = axes[0, 1].bar(severity_labels, severity_counts, color=colors)
        axes[0, 1].set_title("Incident Severity Distribution")
        axes[0, 1].set_ylabel("Count")
        
        for bar, count in zip(bars, severity_counts):
            height = bar.get_height()
            axes[0, 1].text(bar.get_x() + bar.get_width()/2., height + 0.5,
                           str(count), ha='center', va='bottom')
        
        # 3. SLA Compliance Heatmap
        sla_details = report["sla_compliance"]["details"]
        metrics = list(sla_details.keys())
        compliant = [sla_details[m]["compliant"] for m in metrics]
        values = [sla_details[m]["value"] for m in metrics]
        targets = [sla_details[m]["target"] for m in metrics]
        
        x = range(len(metrics))
        width = 0.35
        
        bars1 = axes[1, 0].bar(x, values, width, label='Actual', color=['green' if c else 'red' for c in compliant])
        bars2 = axes[1, 0].bar([i + width for i in x], targets, width, label='Target', color='blue', alpha=0.5)
        
        axes[1, 0].set_title("SLA Compliance")
        axes[1, 0].set_ylabel("Value")
        axes[1, 0].set_xticks([i + width/2 for i in x])
        axes[1, 0].set_xticklabels(metrics, rotation=45, ha='right')
        axes[1, 0].legend()
        
        # 4. Risk Trend
        months = ['Sep', 'Oct', 'Nov', 'Dec', 'Jan']
        # Simulated risk scores
        risk_trend = [65, 52, 48, 41, risk_score]
        
        axes[1, 1].plot(months, risk_trend, marker='o', linewidth=2)
        axes[1, 1].axhline(y=30, color='green', linestyle='--', alpha=0.5, label='Low Risk')
        axes[1, 1].axhline(y=60, color='orange', linestyle='--', alpha=0.5, label='Medium Risk')
        axes[1, 1].axhline(y=80, color='red', linestyle='--', alpha=0.5, label='High Risk')
        
        # Fill between thresholds
        axes[1, 1].fill_between(months, 0, 30, alpha=0.1, color='green')
        axes[1, 1].fill_between(months, 30, 60, alpha=0.1, color='yellow')
        axes[1, 1].fill_between(months, 60, 100, alpha=0.1, color='red')
        
        axes[1, 1].set_title("Risk Score Trend")
        axes[1, 1].set_ylabel("Risk Score (0-100)")
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"Dashboard saved to {output_path}")
        else:
            plt.show()
    
    def _create_risk_gauge(self, ax, value: float, level: str):
        """Create a risk gauge chart."""
        # Set up the gauge
        min_val, max_val = 0, 100
        center = (max_val - min_val) / 2
        
        # Create the gauge background
        angles = np.linspace(0, 180, 100)
        radii = np.ones(100) * 0.5
        
        # Color segments
        green_angle = int(30 * 100 / 180)
        yellow_angle = int(60 * 100 / 180)
        red_angle = 100
        
        # Plot colored segments
        ax.plot(angles[:green_angle], radii[:green_angle], color='green', linewidth=20)
        ax.plot(angles[green_angle:yellow_angle], radii[green_angle:yellow_angle], color='yellow', linewidth=20)
        ax.plot(angles[yellow_angle:], radii[yellow_angle:], color='red', linewidth=20)
        
        # Plot needle
        needle_angle = 180 * (value / 100)
        ax.plot([needle_angle, needle_angle], [0, 0.4], color='black', linewidth=3)
        
        # Add center circle
        ax.plot(0, 0, 'ko', markersize=10)
        
        # Set limits and remove axes
        ax.set_xlim(-0.1, 180.1)
        ax.set_ylim(0, 0.6)
        ax.axis('off')
        
        # Add labels
        ax.text(90, 0.7, f"Risk Score: {value:.1f}", ha='center', fontsize=12, fontweight='bold')
        ax.text(90, 0.8, f"Level: {level}", ha='center', fontsize=14, 
               color={'LOW': 'green', 'MEDIUM': 'orange', 'HIGH': 'red'}.get(level, 'black'),
               fontweight='bold')
        
        # Add threshold labels
        ax.text(0, -0.1, "0", ha='center', fontsize=10)
        ax.text(90, -0.1, "50", ha='center', fontsize=10)
        ax.text(180, -0.1, "100", ha='center', fontsize=10)
        
        ax.set_title("Risk Exposure Gauge", fontsize=12)
    
    def generate_executive_summary(self, tenant_id: str, output_path: str = None):
        """Generate comprehensive executive summary."""
        report = self.metrics_system.generate_executive_report(tenant_id)
        
        summary = f"""
        {'='*80}
        EXECUTIVE SUMMARY - {tenant_id.upper()}
        {'='*80}
        
        📊 FINANCIAL IMPACT (Last 30 Days):
        • Total Savings: ${report['financial_metrics']['total_cost_savings']:,.2f}
        • ROI: {report['financial_metrics']['estimated_roi']:.1f}%
        • Revenue Impact Prevented: ${report['financial_metrics']['revenue_impact_prevented']:,.2f}
        
        ⚙️ OPERATIONAL EFFICIENCY:
        • MTTR: {report['operational_metrics']['mttr_minutes']:.1f} minutes
          (Improvement: {report['operational_metrics']['mttr_improvement_percent']:.1f}%)
        • Incidents Auto-Resolved: {report['operational_metrics']['incidents_auto_resolved']}
        • Automation Rate: {report['operational_metrics']['automation_rate_percent']:.1f}%
        • Engineer Hours Saved: {report['operational_metrics']['engineer_hours_saved']:.0f}
        
        🛡️ RISK MANAGEMENT:
        • Risk Exposure: {report['risk_metrics']['risk_exposure_score']:.1f}/100
          (Level: {report['risk_metrics']['risk_level']})
        • Catastrophic Failures: {report['risk_metrics']['catastrophic_failures']}
        • SLA Violations: {report['risk_metrics']['sla_violations']}
        
        📈 BUSINESS IMPACT:
        • Engineer Efficiency Gain: {report['business_impact']['engineer_efficiency_gain']:.1f}%
        • Downtime Reduction: {report['business_impact']['downtime_reduction_percent']:.1f}%
        • Customer Satisfaction: {report['business_impact']['customer_satisfaction_impact']}
        • Operational Risk Reduction: {report['business_impact']['operational_risk_reduction']}
        
        📋 SLA COMPLIANCE:
        • Overall Status: {'✅ COMPLIANT' if report['sla_compliance']['overall_compliant'] else '❌ NON-COMPLIANT'}
        • Penalties Incurred: ${report['sla_compliance']['penalties_incurred']:.2f}
        
        {'='*80}
        """
        
        if output_path:
            with open(output_path, 'w') as f:
                f.write(summary)
            print(f"Summary saved to {output_path}")
        
        return summary


# Example usage
if __name__ == "__main__":
    from metrics_system import EnterpriseMetricsSystem
    
    # Initialize systems
    metrics_system = EnterpriseMetricsSystem()
    dashboard = ExecutiveDashboard(metrics_system)
    
    # Generate dashboards
    dashboard.generate_financial_dashboard("acme-corp", "financial_dashboard.png")
    dashboard.generate_risk_dashboard("acme-corp", "risk_dashboard.png")
    
    # Generate executive summary
    summary = dashboard.generate_executive_summary("acme-corp", "executive_summary.txt")
    print(summary)

