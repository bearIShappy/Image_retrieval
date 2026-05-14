import { type Analytics } from '../api';
import { Activity, Clock, Target, Hash, Cpu } from 'lucide-react';
import { motion } from 'framer-motion';

interface AnalyticsPanelProps {
  analytics: Analytics | null;
}

export function AnalyticsPanel({ analytics }: AnalyticsPanelProps) {
  if (!analytics) return null;

  const stats = [
    {
      label: 'Retrieval Time',
      value: `${analytics.time_ms}ms`,
      icon: <Clock className="w-5 h-5 text-accent" />,
      color: 'border-blue-500/30 bg-blue-500/10'
    },
    {
      label: 'Mode',
      value: analytics.mode,
      icon: <Cpu className="w-5 h-5 text-purple-400" />,
      color: 'border-purple-500/30 bg-purple-500/10'
    },
    {
      label: 'Top Score',
      value: `${(analytics.top_score * 100).toFixed(1)}%`,
      icon: <Target className="w-5 h-5 text-green-400" />,
      color: 'border-green-500/30 bg-green-500/10'
    },
    {
      label: 'Retrieved',
      value: analytics.total_retrieved,
      icon: <Hash className="w-5 h-5 text-orange-400" />,
      color: 'border-orange-500/30 bg-orange-500/10'
    }
  ];

  return (
    <motion.div 
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className="flex flex-col gap-4 w-full mt-8"
    >
      <div className="flex items-center gap-2">
        <Activity className="w-5 h-5 text-primary-400" />
        <h2 className="text-xl font-semibold">Performance Analytics</h2>
      </div>

      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
        {stats.map((stat, idx) => (
          <div key={idx} className="glass-card p-4 rounded-2xl flex items-center gap-4 hover:bg-surface-hover transition-colors">
            <div className={`p-3 rounded-xl border ${stat.color}`}>
              {stat.icon}
            </div>
            <div className="flex flex-col">
              <span className="text-sm font-medium text-text-tertiary">{stat.label}</span>
              <span className="text-xl font-bold text-text-primary capitalize">{stat.value}</span>
            </div>
          </div>
        ))}
      </div>
    </motion.div>
  );
}
