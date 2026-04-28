# React + TypeScript + Vite

이 템플릿은 HMR 과 일부 ESLint 규칙이 있는 Vite 에서 React 를 동작시키는
최소 셋업을 제공합니다.

현재 두 개의 공식 플러그인이 사용 가능:

- [@vitejs/plugin-react](https://github.com/vitejs/vite-plugin-react/blob/main/packages/plugin-react) 는 [Oxc](https://oxc.rs) 를 사용
- [@vitejs/plugin-react-swc](https://github.com/vitejs/vite-plugin-react/blob/main/packages/plugin-react-swc) 는 [SWC](https://swc.rs/) 를 사용

## React Compiler

React Compiler 는 dev/build 성능에 영향을 주기 때문에 이 템플릿에서는
활성화되어 있지 않습니다. 추가하려면 [이 문서](https://react.dev/learn/react-compiler/installation)
를 참조하세요.

## ESLint 설정 확장

프로덕션 애플리케이션을 개발하고 있다면, 타입 인식 lint 규칙을 활성화
하기 위해 설정을 업데이트하기를 권장합니다:

```js
export default defineConfig([
  globalIgnores(['dist']),
  {
    files: ['**/*.{ts,tsx}'],
    extends: [
      // 다른 설정...

      // tseslint.configs.recommended 를 제거하고 이것으로 교체
      tseslint.configs.recommendedTypeChecked,
      // 또는 더 엄격한 규칙을 위해 이것 사용
      tseslint.configs.strictTypeChecked,
      // 선택적으로 stylistic 규칙을 위해 이것 추가
      tseslint.configs.stylisticTypeChecked,

      // 다른 설정...
    ],
    languageOptions: {
      parserOptions: {
        project: ['./tsconfig.node.json', './tsconfig.app.json'],
        tsconfigRootDir: import.meta.dirname,
      },
      // 다른 옵션...
    },
  },
])
```

React 전용 lint 규칙을 위해 [eslint-plugin-react-x](https://github.com/Rel1cx/eslint-react/tree/main/packages/plugins/eslint-plugin-react-x) 와 [eslint-plugin-react-dom](https://github.com/Rel1cx/eslint-react/tree/main/packages/plugins/eslint-plugin-react-dom) 도 설치할 수 있습니다:

```js
// eslint.config.js
import reactX from 'eslint-plugin-react-x'
import reactDom from 'eslint-plugin-react-dom'

export default defineConfig([
  globalIgnores(['dist']),
  {
    files: ['**/*.{ts,tsx}'],
    extends: [
      // 다른 설정...
      // React 용 lint 규칙 활성화
      reactX.configs['recommended-typescript'],
      // React DOM 용 lint 규칙 활성화
      reactDom.configs.recommended,
    ],
    languageOptions: {
      parserOptions: {
        project: ['./tsconfig.node.json', './tsconfig.app.json'],
        tsconfigRootDir: import.meta.dirname,
      },
      // 다른 옵션...
    },
  },
])
```
