import { redirect } from 'next/navigation'

export default function Home() {
  // Single user app - go straight to the app
  redirect('/app')
}
