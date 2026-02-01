import typescriptEslintPlugin from '@typescript-eslint/eslint-plugin'
import typescriptParser from '@typescript-eslint/parser'
import reactHooksPlugin from 'eslint-plugin-react-hooks'

/** @type {import('eslint').Linter.Config[]} */
const eslintConfig = [
  {
    ignores: ['node_modules/', '.next/', 'dist/', '.turbo/', '**/*schemas.ts'],
  },
  {
    files: ['**/*.{js,jsx,ts,tsx}'],
    languageOptions: {
      parser: typescriptParser,
      sourceType: 'module',
      ecmaVersion: 2021,
      parserOptions: {
        ecmaFeatures: {
          jsx: true,
        },
      },
    },
    plugins: {
      '@typescript-eslint': typescriptEslintPlugin,
      'react-hooks': reactHooksPlugin,
    },
    rules: {
      '@typescript-eslint/no-unused-vars': [
        'error',
        {
          argsIgnorePattern: '^_',
          varsIgnorePattern: '^_',
          caughtErrorsIgnorePattern: '^_',
        },
      ],
      'react-hooks/exhaustive-deps': 'warn',
      semi: ['error', 'never'],
      '@next/next/no-img-element': 'off', // Disable Next.js Image enforcement
    },
  },
]

export default eslintConfig
