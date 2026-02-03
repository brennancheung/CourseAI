import { redirect } from 'next/navigation'

export default function AppPage() {
  // Redirect to the first lesson in the curriculum
  redirect('/app/lesson/what-is-learning')
}
