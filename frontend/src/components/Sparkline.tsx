interface Props {
  data: number[];
  color?: string;
  height?: number;
  width?: number;
  /** Make the SVG fill its container; use for contained KPI cards. */
  responsive?: boolean;
}

export function Sparkline({ data, color = "#16A34A", height = 44, width = 140, responsive = false }: Props) {
  if (!data.length) return null;
  const min = Math.min(...data), max = Math.max(...data);
  const range = max - min || 1;
  const stepX = width / (data.length - 1);
  const pts = data
    .map((v, i) => `${i * stepX},${height - ((v - min) / range) * (height - 6) - 3}`)
    .join(" ");

  if (responsive) {
    return (
      <svg
        viewBox={`0 0 ${width} ${height}`}
        preserveAspectRatio="none"
        className="block w-full h-full"
      >
        <polyline fill="none" stroke={color} strokeWidth={1.5} strokeLinecap="round" strokeLinejoin="round" points={pts} />
      </svg>
    );
  }

  return (
    <svg width={width} height={height} className="overflow-visible">
      <polyline fill="none" stroke={color} strokeWidth={2} strokeLinecap="round" strokeLinejoin="round" points={pts} />
      {data.map((v, i) => (
        <circle key={i} cx={i * stepX} cy={height - ((v - min) / range) * (height - 6) - 3} r={i === data.length - 1 ? 3 : 0} fill={color} />
      ))}
    </svg>
  );
}
