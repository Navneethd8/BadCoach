import '@fontsource-variable/material-symbols-outlined/standard.css'
import { usePageSeo } from './seo/usePageSeo'
import AnalyzePageContent from './components/AnalyzePageContent'
import { useAnalyzeController } from './useAnalyzeController'

export default function App() {
    usePageSeo('/analyze')
    const controller = useAnalyzeController()

    return <AnalyzePageContent controller={controller} />
}
