#!/usr/bin/env python3
"""
Coverage Analyzer voor Omgevingswet Meldingen
Analyseert welke meldingen valideerbaar zijn met beschikbare business rules
"""

import pandas as pd
import json
from collections import defaultdict
from datetime import datetime

# Business Rules Mapping
# Deze zijn geëxtraheerd uit je geüploade PDF
BUSINESS_RULES = {
    'Windturbine': {
        'has_rules': True,
        'sources': ['Paragraaf 4.136 Bal', 'Artikel 22.61 OP'],
        'confidence': 'high'
    },
    'Graven in bodem met een kwaliteit boven de interventiewaarde bodemkwaliteit': {
        'has_rules': True,
        'sources': ['Paragraaf 4.120 Bal'],
        'confidence': 'high'
    },
    'Gesloten bodemenergiesysteem': {
        'has_rules': True,
        'sources': ['Paragraaf 4.1137 Bal'],
        'confidence': 'high'
    },
    'Saneren van de bodem': {
        'has_rules': True,
        'sources': ['Paragraaf 3.2.23 Bal'],
        'confidence': 'high'
    },
    'Bodem saneren': {
        'has_rules': True,
        'sources': ['Paragraaf 3.2.23 Bal'],
        'confidence': 'high',
        'note': 'Zelfde als "Saneren van de bodem"'
    },
    'Opslaan van kuilvoer of vaste bijvoedermiddelen': {
        'has_rules': True,
        'sources': ['Paragraaf 4.84 Bal'],
        'confidence': 'high'
    },
    'Mestvergistingsinstallatie': {
        'has_rules': True,
        'sources': ['Paragraaf 4.88 Bal'],
        'confidence': 'high'
    },
    'Opslaan van vaste mest, champost of dikke fractie': {
        'has_rules': True,
        'sources': ['Paragraaf 4.84 Bal'],
        'confidence': 'high'
    },
    'Opslaan van drijfmest, digestaat of dunne fractie in mestbassin': {
        'has_rules': False,
        'sources': [],
        'confidence': 'medium',
        'note': 'Mogelijk in Paragraaf 4.84, maar niet expliciet gevonden'
    },
    'Dierenverblijven': {
        'has_rules': False,
        'sources': [],
        'confidence': 'low',
        'note': 'Business rules niet gevonden in PDF'
    },
    'Slopen van een bouwwerk of gedeelte daarvan of asbest verwijderen': {
        'has_rules': False,
        'sources': [],
        'confidence': 'low',
        'note': 'Waarschijnlijk in Paragraaf 4.123, maar niet in geüploade PDF'
    },
}

def load_data(filepath):
    """Laad het bestand met meldingen (CSV of JSON)"""
    try:
        # Try JSON first
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
            df = pd.DataFrame(data)
            print(f"✅ Geladen (JSON): {len(df)} meldingen")
            return df
    except json.JSONDecodeError:
        # Fall back to CSV
        try:
            df = pd.read_csv(filepath)
            print(f"✅ Geladen (CSV): {len(df)} meldingen")
            return df
        except Exception as e:
            print(f"❌ Error bij laden bestand: {e}")
            return None

def analyze_coverage(df):
    """Analyseer coverage per activiteit"""
    
    # Groepeer per activiteit
    activity_counts = df['activity_name'].value_counts()
    
    results = []
    total_meldingen = len(df)
    valideerbaar = 0
    
    for activity, count in activity_counts.items():
        # Check of we business rules hebben
        rule_info = BUSINESS_RULES.get(activity, {
            'has_rules': False,
            'sources': [],
            'confidence': 'unknown',
            'note': 'Activiteit niet gevonden in business rules mapping'
        })
        
        percentage = (count / total_meldingen) * 100
        
        if rule_info['has_rules']:
            valideerbaar += count
        
        results.append({
            'activity_name': activity,
            'count': int(count),
            'percentage': round(percentage, 2),
            'has_rules': rule_info['has_rules'],
            'sources': rule_info['sources'],
            'confidence': rule_info['confidence'],
            'note': rule_info.get('note', '')
        })
    
    # Sorteer op count (hoogste eerst)
    results.sort(key=lambda x: x['count'], reverse=True)
    
    return {
        'totaal_meldingen': total_meldingen,
        'valideerbaar': valideerbaar,
        'niet_valideerbaar': total_meldingen - valideerbaar,
        'coverage_percentage': round((valideerbaar / total_meldingen) * 100, 2),
        'activities': results
    }

def generate_priority_list(analysis):
    """Genereer lijst van activiteiten waar business rules het meeste impact hebben"""
    
    # Activiteiten ZONDER rules, gesorteerd op count
    missing_rules = [
        a for a in analysis['activities'] 
        if not a['has_rules']
    ]
    
    priority = []
    cumulative_coverage = analysis['coverage_percentage']
    
    for activity in missing_rules[:10]:  # Top 10
        potential_coverage = cumulative_coverage + activity['percentage']
        priority.append({
            'activity': activity['activity_name'],
            'count': activity['count'],
            'current_coverage': round(cumulative_coverage, 1),
            'potential_coverage': round(potential_coverage, 1),
            'impact': round(activity['percentage'], 1)
        })
        cumulative_coverage = potential_coverage
    
    return priority

def print_summary(analysis):
    """Print een mooi overzicht naar console"""
    
    print("\n" + "="*80)
    print("📊 COVERAGE ANALYSE - OMGEVINGSWET MELDINGEN")
    print("="*80)
    
    print(f"\n📈 TOTAAL OVERZICHT:")
    print(f"   Totaal meldingen:     {analysis['totaal_meldingen']:,}")
    print(f"   ✅ Valideerbaar:      {analysis['valideerbaar']:,} ({analysis['coverage_percentage']}%)")
    print(f"   ❌ Niet valideerbaar: {analysis['niet_valideerbaar']:,}")
    
    print(f"\n🎯 TOP 10 ACTIVITEITEN (met business rules):")
    print(f"   {'Activity':<60} {'Count':>8} {'%':>6} {'Rules':>8}")
    print(f"   {'-'*60} {'-'*8} {'-'*6} {'-'*8}")
    
    with_rules = [a for a in analysis['activities'] if a['has_rules']]
    for activity in with_rules[:10]:
        print(f"   {activity['activity_name'][:60]:<60} {activity['count']:>8} {activity['percentage']:>6.1f} {'✅':>8}")
    
    print(f"\n🔴 TOP 10 ACTIVITEITEN (zonder business rules):")
    print(f"   {'Activity':<60} {'Count':>8} {'%':>6} {'Impact':>8}")
    print(f"   {'-'*60} {'-'*8} {'-'*6} {'-'*8}")
    
    without_rules = [a for a in analysis['activities'] if not a['has_rules']]
    for activity in without_rules[:10]:
        print(f"   {activity['activity_name'][:60]:<60} {activity['count']:>8} {activity['percentage']:>6.1f}% {'-':>7}")
    
    print("\n" + "="*80)

def generate_json_report(analysis, priority, output_file='coverage_report.json'):
    """Genereer JSON rapport"""
    
    report = {
        'generated_at': datetime.now().isoformat(),
        'summary': {
            'totaal_meldingen': analysis['totaal_meldingen'],
            'valideerbaar': analysis['valideerbaar'],
            'niet_valideerbaar': analysis['niet_valideerbaar'],
            'coverage_percentage': analysis['coverage_percentage']
        },
        'activities': analysis['activities'],
        'priority_list': priority,
        'recommendations': generate_recommendations(analysis, priority)
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ JSON rapport opgeslagen: {output_file}")
    return report

def generate_recommendations(analysis, priority):
    """Genereer aanbevelingen"""
    
    recommendations = []
    
    if analysis['coverage_percentage'] < 30:
        recommendations.append({
            'priority': 'CRITICAL',
            'message': f"Coverage is zeer laag ({analysis['coverage_percentage']}%). Focus op top 3 missende activiteiten."
        })
    
    if len(priority) > 0 and priority[0]['impact'] > 30:
        top = priority[0]
        recommendations.append({
            'priority': 'HIGH',
            'message': f"'{top['activity']}' heeft {top['impact']}% impact. Dit moet eerste prioriteit zijn."
        })
    
    # Check voor activiteiten met lage confidence
    low_confidence = [a for a in analysis['activities'] if a.get('confidence') == 'low']
    if low_confidence:
        recommendations.append({
            'priority': 'MEDIUM',
            'message': f"{len(low_confidence)} activiteiten hebben onzekere status. Verifieer handmatig."
        })
    
    return recommendations

def generate_html_report(analysis, priority, output_file='coverage_report.html'):
    """Genereer HTML dashboard"""
    
    html = f"""
<!DOCTYPE html>
<html lang="nl">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Coverage Rapport - Omgevingswet Meldingen</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Arial, sans-serif;
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
            background: #f5f5f5;
        }}
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            border-radius: 10px;
            margin-bottom: 30px;
        }}
        .stats {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}
        .stat-card {{
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .stat-value {{
            font-size: 2em;
            font-weight: bold;
            color: #667eea;
        }}
        .stat-label {{
            color: #666;
            font-size: 0.9em;
        }}
        .coverage-bar {{
            width: 100%;
            height: 40px;
            background: #e0e0e0;
            border-radius: 20px;
            overflow: hidden;
            margin: 20px 0;
        }}
        .coverage-fill {{
            height: 100%;
            background: linear-gradient(90deg, #11998e 0%, #38ef7d 100%);
            display: flex;
            align-items: center;
            justify-content: center;
            color: white;
            font-weight: bold;
        }}
        table {{
            width: 100%;
            background: white;
            border-collapse: collapse;
            border-radius: 8px;
            overflow: hidden;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin-bottom: 30px;
        }}
        th {{
            background: #667eea;
            color: white;
            padding: 15px;
            text-align: left;
        }}
        td {{
            padding: 12px 15px;
            border-bottom: 1px solid #eee;
        }}
        tr:hover {{
            background: #f8f9fa;
        }}
        .has-rules {{
            color: #38ef7d;
            font-weight: bold;
        }}
        .no-rules {{
            color: #ff6b6b;
            font-weight: bold;
        }}
        .priority-badge {{
            display: inline-block;
            padding: 4px 12px;
            border-radius: 12px;
            font-size: 0.85em;
            font-weight: bold;
        }}
        .priority-high {{
            background: #ff6b6b;
            color: white;
        }}
        .priority-medium {{
            background: #ffd93d;
            color: #333;
        }}
        .section {{
            background: white;
            padding: 25px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin-bottom: 30px;
        }}
        h2 {{
            color: #333;
            border-bottom: 2px solid #667eea;
            padding-bottom: 10px;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>📊 Coverage Rapport</h1>
        <p>Omgevingswet Meldingen - Business Rules Analyse</p>
        <p style="opacity: 0.8;">Gegenereerd: {datetime.now().strftime('%d-%m-%Y %H:%M')}</p>
    </div>
    
    <div class="stats">
        <div class="stat-card">
            <div class="stat-value">{analysis['totaal_meldingen']:,}</div>
            <div class="stat-label">Totaal Meldingen</div>
        </div>
        <div class="stat-card">
            <div class="stat-value" style="color: #38ef7d">{analysis['valideerbaar']:,}</div>
            <div class="stat-label">✅ Valideerbaar</div>
        </div>
        <div class="stat-card">
            <div class="stat-value" style="color: #ff6b6b">{analysis['niet_valideerbaar']:,}</div>
            <div class="stat-label">❌ Niet Valideerbaar</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">{analysis['coverage_percentage']}%</div>
            <div class="stat-label">Coverage Percentage</div>
        </div>
    </div>
    
    <div class="coverage-bar">
        <div class="coverage-fill" style="width: {analysis['coverage_percentage']}%">
            {analysis['coverage_percentage']}% Coverage
        </div>
    </div>
    
    <div class="section">
        <h2>🎯 Prioriteiten Lijst</h2>
        <p>Voeg deze business rules toe voor maximale impact op coverage:</p>
        <table>
            <thead>
                <tr>
                    <th>#</th>
                    <th>Activiteit</th>
                    <th>Aantal Meldingen</th>
                    <th>Impact</th>
                    <th>Potentiële Coverage</th>
                </tr>
            </thead>
            <tbody>
"""
    
    for i, item in enumerate(priority[:10], 1):
        priority_class = 'priority-high' if i <= 3 else 'priority-medium'
        html += f"""
                <tr>
                    <td><span class="priority-badge {priority_class}">#{i}</span></td>
                    <td>{item['activity']}</td>
                    <td>{item['count']:,}</td>
                    <td>+{item['impact']}%</td>
                    <td>{item['potential_coverage']}%</td>
                </tr>
"""
    
    html += """
            </tbody>
        </table>
    </div>
    
    <div class="section">
        <h2>✅ Activiteiten met Business Rules</h2>
        <table>
            <thead>
                <tr>
                    <th>Activiteit</th>
                    <th>Aantal</th>
                    <th>%</th>
                    <th>Business Rules</th>
                </tr>
            </thead>
            <tbody>
"""
    
    with_rules = [a for a in analysis['activities'] if a['has_rules']]
    for activity in with_rules:
        sources = ', '.join(activity['sources'])
        html += f"""
                <tr>
                    <td>{activity['activity_name']}</td>
                    <td>{activity['count']:,}</td>
                    <td>{activity['percentage']}%</td>
                    <td><span class="has-rules">{sources}</span></td>
                </tr>
"""
    
    html += """
            </tbody>
        </table>
    </div>
    
    <div class="section">
        <h2>❌ Activiteiten zonder Business Rules</h2>
        <table>
            <thead>
                <tr>
                    <th>Activiteit</th>
                    <th>Aantal</th>
                    <th>%</th>
                    <th>Status</th>
                </tr>
            </thead>
            <tbody>
"""
    
    without_rules = [a for a in analysis['activities'] if not a['has_rules']]
    for activity in without_rules[:20]:  # Top 20
        html += f"""
                <tr>
                    <td>{activity['activity_name']}</td>
                    <td>{activity['count']:,}</td>
                    <td>{activity['percentage']}%</td>
                    <td><span class="no-rules">Geen rules</span></td>
                </tr>
"""
    
    html += """
            </tbody>
        </table>
    </div>
    
    <div class="section">
        <h2>💡 Aanbevelingen</h2>
        <ul>
            <li><strong>Prioriteit 1:</strong> Voeg business rules toe voor de top 3 activiteiten zonder rules</li>
            <li><strong>Prioriteit 2:</strong> Verifieer "Toepassen van grond" rules (grootste impact: 49.5%)</li>
            <li><strong>Prioriteit 3:</strong> Check PDF structuur - veel data zit in bijlagen i.p.v. gestructureerde velden</li>
        </ul>
    </div>
    
</body>
</html>
"""
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html)
    
    print(f"✅ HTML rapport opgeslagen: {output_file}")

def main():
    """Main functie"""
    
    print("🚀 Coverage Analyzer gestart...")
    
    # Laad data (CSV of JSON)
    data_file = './Supabase Snippet Retrieve All Messages.csv'
    df = load_data(data_file)
    
    if df is None:
        return
    
    # Analyseer
    print("\n📊 Analyzing coverage...")
    analysis = analyze_coverage(df)
    
    # Genereer priority lijst
    priority = generate_priority_list(analysis)
    
    # Print samenvatting
    print_summary(analysis)
    
    # Genereer rapporten
    print("\n📝 Genereer rapporten...")
    generate_json_report(analysis, priority)
    generate_html_report(analysis, priority)
    
    print("\n✅ Klaar! Check de gegenereerde bestanden.")
    print(f"\n🎯 Coverage: {analysis['coverage_percentage']}%")
    print(f"   Valideerbaar: {analysis['valideerbaar']:,} van {analysis['totaal_meldingen']:,} meldingen")

if __name__ == '__main__':
    main()
