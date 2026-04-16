package com.aipipeline.coreapi.common.web;

import com.aipipeline.coreapi.common.InternalSecretProperties;
import org.springframework.context.annotation.Configuration;
import org.springframework.web.servlet.config.annotation.InterceptorRegistry;
import org.springframework.web.servlet.config.annotation.WebMvcConfigurer;

/**
 * Registers the shared-secret interceptor for all internal endpoints.
 */
@Configuration
public class InternalWebConfig implements WebMvcConfigurer {

    private final InternalSecretProperties secretProperties;

    public InternalWebConfig(InternalSecretProperties secretProperties) {
        this.secretProperties = secretProperties;
    }

    @Override
    public void addInterceptors(InterceptorRegistry registry) {
        registry.addInterceptor(new InternalSecretAuthInterceptor(secretProperties))
                .addPathPatterns("/api/internal/**");
    }
}
