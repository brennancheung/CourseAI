interface YouTubeEmbedProps {
  videoId: string
  title?: string
}

export function YouTubeEmbed({ videoId, title }: YouTubeEmbedProps) {
  return (
    <div className="relative w-full aspect-video rounded-lg overflow-hidden border bg-black">
      <iframe
        src={`https://www.youtube.com/embed/${videoId}`}
        title={title ?? 'YouTube video'}
        allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
        allowFullScreen
        className="absolute inset-0 w-full h-full"
      />
    </div>
  )
}
