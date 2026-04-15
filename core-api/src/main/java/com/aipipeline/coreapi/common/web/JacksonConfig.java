package com.aipipeline.coreapi.common.web;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.SerializationFeature;
import com.fasterxml.jackson.datatype.jsr310.JavaTimeModule;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

/**
 * Explicit Jackson configuration.
 *
 * Spring Boot 4.0 reorganized the Jackson auto-configuration and no longer
 * publishes a singleton {@link ObjectMapper} bean for arbitrary injection
 * from a bare {@code spring-boot-starter-web}. We depend on that bean in
 * {@code RedisJobDispatchAdapter} (and potentially elsewhere), so we
 * declare it ourselves with the JavaTimeModule registered so {@link
 * java.time.Instant} serializes as an ISO-8601 string rather than a
 * numeric timestamp.
 */
@Configuration
public class JacksonConfig {

    @Bean
    public ObjectMapper objectMapper() {
        ObjectMapper mapper = new ObjectMapper();
        mapper.registerModule(new JavaTimeModule());
        mapper.disable(SerializationFeature.WRITE_DATES_AS_TIMESTAMPS);
        return mapper;
    }
}
